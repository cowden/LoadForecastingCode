import numpy as np
import pandas as pd
import os
import sys
import json
from sys import path
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pdb, traceback
import datetime
import time

# from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# Figure
import matplotlib.pyplot as plt



# Options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', 50000)



#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def ParameterSetting():
    # Cross Validation Parameters
    Param={'CVN': 3, 'CVP':0.1, 'CVOptExternal':"HoldOut"}

    # Machine List
    Param['MachineList']=['RF']   # LogReg KNN GBM 'LogReg', ,'Ridge' "Reg"

    # Group Number
    Param['NGroup']=[1]

    # Ensemble Option
    Param['Ensem']="Simple"

    # Internal Cross Validation Option
    Param['CVOptInternal']="HoldOut"

    # GBM
    Param['GBM_NTry']=1000
    Param['GBM_LRate']=0.01
    Param['GBM_CVP']=0.1

    # Random Forest
    Param['RF_NTree']=500
    Param['RF_CVP']=0.1

    # SVM
    Param['SVM_CVP']=0.1
    Param['C']=[1] # [3,5,7,8,9,10]
    Param['Gamma']=0.5
#
#	# NuSVM
#	Param['SVM_CVP']=0.1
#	Param['Nu']=[0.0001, 0.0002, 0.0003, 0.0005, 0.001]
#
#	# KNN
#	Param['NNeighbors']=3000
#
    # Ridge Regression
    Param['Reg_Reg']=[0.0001, 0.001, 0.01,0.1,0.5,1,5,10,100,1000]
    Param['Reg_CVN']=5
    Param['Reg_CVP']=0.1

    # Number of Class
    Param['Min']=0
    Param['Max']=700000
#
#	# XGBoost Parameters
#	params = {}
#	params["objective"] = "reg:linear"
#	params["eta"] = 0.05
#	params["min_child_weight"] = 5
#	params["subsample"] = 0.8
#	params["colsample_bytree"] = 0.8
#	params["scale_pos_weight"] = 1.0
#	params["silent"] = 1
#	params["max_depth"] = 9
#	plst = list(params.items())
#	Param['XGB']=plst
#	Param['XGB_NR']=300   # 300


    return Param



#######################################################################
## Old Price and Weather Reader
#######################################################################
def OldWeatherPriceRead(OldPriceLoc, OldPriceFile, OldWeatherLoc, OldWeatherFile):

    OldWeather = pd.read_csv(OldWeatherLoc + OldWeatherFile)
    OldPrice = pd.read_csv(OldPriceLoc + OldPriceFile)
    OldPrice ["Date"]=pd.to_datetime(OldPrice["Date"])

    SingleWeatherStation = OldWeather[ OldWeather.point==1]
    del SingleWeatherStation["point"]
    for t in range(2,19):
        Temp2 = OldWeather[ OldWeather.point==t]
        del Temp2["point"]
        SingleWeatherStation = pd.merge(SingleWeatherStation,Temp2,
                                        on=["Date","Year","Month","Day","Hour"],suffixes=('',str(t)))

    SingleWeatherStation["Ahead"] = 1
    SingleWeatherStation["Date"]=pd.to_datetime(SingleWeatherStation["Date"])

    ##
    OldWeatherPrice=pd.merge(SingleWeatherStation,OldPrice,on=["Date","Year","Month","Day","Hour"])
    return OldWeatherPrice



#######################################################################
## Price Reader
#######################################################################
def PriceReader(PriceFileLoc,PriceFileName):
    RawPrice  = pd.read_csv(PriceFileLoc + PriceFileName)
    RawPrice["Date"]=pd.to_datetime(RawPrice["Date"])
    RawPrice.rename(columns={'Date': 'prediction_date'},inplace=True)
    return RawPrice



#######################################################################
## Weather Data Reading
#######################################################################
def SingleWeather(WeatherFileLoc):
    FileList=os.listdir(WeatherFileLoc)
    FileList.sort()

    ##
    for f in range(0,len(FileList)):
        ReadData  = pd.read_csv(WeatherFileLoc + FileList[f])

        SingleWeatherStation = ReadData[ ReadData.point==1]
        del SingleWeatherStation["point"]
        for t in range(2,19):  # 19
            Temp2 = ReadData[ ReadData.point==t]
            del Temp2["point"]
            SingleWeatherStation = pd.merge(SingleWeatherStation,Temp2,on=TimeIndex,suffixes=('',str(t)))

        SingleWeatherStation["Ahead"] = Ahead
        if f==0:
            Data = SingleWeatherStation
        else:
            Data = pd.concat([Data,SingleWeatherStation])

    Data.index=range(0,Data.shape[0])
    Data["available_date2"]=Data[TimeIndex[0]].str.replace('06:00:00','23:00:00')
    Data["available_date2"]=Data["available_date2"].str.replace('6:00','23:00:00')
    Data["available_date2"]=pd.to_datetime(Data["available_date2"])
    Data[TimeIndex[0]]=pd.to_datetime(Data[TimeIndex[0]])
    Data[TimeIndex[1]]=pd.to_datetime(Data[TimeIndex[1]])

    return Data




#######################################################################
## Generation Reader
#######################################################################
def PTGenerationReader(PTGenerationLoc,PTGenerationFileName):
    PTGen = pd.read_csv(PTGenerationLoc + PTGenerationFileName)
    PTGen["Date"]=pd.to_datetime(PTGen["Date"])
    PTGen['Gen'] = PTGen.iloc[:,range(6,19)].sum(axis=1)

    TimeIndexPT=["Date","Year","Month","Day","Hour"]
    PTAverageGen = pd.pivot_table(PTGen,values=["demand","Gen"],index=TimeIndexPT,aggfunc=np.mean)
    PTAverageGen=pd.DataFrame(PTAverageGen).reset_index()
    PTAverageGen.rename(columns={'demand': 'Demand'}, inplace=True)
    PTAverageGen.rename(columns={'Date': 'prediction_date'},inplace=True)
    del PTAverageGen["Gen"]
    return PTAverageGen



#######################################################################
## Generation Reader
#######################################################################
def ESGenerationReader(PTGenerationLoc,PTGenerationFileName):
    PTGen = pd.read_csv(PTGenerationLoc + PTGenerationFileName)
    PTGen["Date"]=pd.to_datetime(PTGen["Date"])
    PTGen['Gen'] = PTGen.iloc[:,range(6,20)].sum(axis=1)

    TimeIndexPT=["Date","Year","Month","Day","Hour"]
    PTAverageGen = pd.pivot_table(PTGen,values=["forecast_demand","Gen"],index=TimeIndexPT,aggfunc=np.mean)
    PTAverageGen=pd.DataFrame(PTAverageGen).reset_index()
    PTAverageGen.rename(columns={'forecast_demand': 'Demand'}, inplace=True)
    PTAverageGen.rename(columns={'Date': 'prediction_date'},inplace=True)
    del PTAverageGen["Gen"]
    return PTAverageGen




########################################################################
# Training Extention
########################################################################
def TrainingExtention(Selection, TotalTrainX, TotalTestX):



    TotalTrainX2=TotalTrainX.iloc[:,Selection]
    TotalTestX2=TotalTestX.iloc[:,Selection]

    TotalTrainX1=pd.concat([TotalTrainX2.iloc[0:1,:], TotalTrainX2.iloc[:-1,:]],axis=0)
    TotalTrainX1.index=range(0,TotalTrainX1.shape[0])
    TotalTrainX1.rename(columns={'Ahead': 'Ahead2',
                                 'Hour': 'Hour2'},inplace=True)
    TotalTrainX3=pd.concat([TotalTrainX2.iloc[1:,:], TotalTrainX2.iloc[167:168,:]],axis=0)
    TotalTrainX3.index=range(0,TotalTrainX3.shape[0])
    TotalTrainX3.rename(columns={'Ahead': 'Ahead2',
                                 'Hour': 'Hour2'},inplace=True)

    TotalTrainX=pd.concat([TotalTrainX1, TotalTrainX2,TotalTrainX3],axis=1)
    LocAhead=np.where(TotalTrainX.columns=="Ahead")[0][0]
    LocHour=np.where(TotalTrainX.columns=="Hour")[0][0]
    TotalTrainX.columns=range(0,TotalTrainX.shape[1])
    TotalTrainX.rename(columns={TotalTrainX.columns[LocAhead]:"Ahead",
                                TotalTrainX.columns[LocHour]:"Hour"},inplace=True)


    TotalTestX1=pd.concat([TotalTestX2.iloc[0:1,:], TotalTestX2.iloc[:-1,:]],axis=0)
    TotalTestX1.index=range(0,TotalTestX1.shape[0])
    TotalTestX1.rename(columns={'Ahead': 'Ahead2',
                                 'Hour': 'Hour2'},inplace=True)
    TotalTestX3=pd.concat([TotalTestX2.iloc[1:,:], TotalTestX2.iloc[167:168,:]],axis=0)
    TotalTestX3.index=range(0,TotalTestX3.shape[0])
    TotalTestX3.rename(columns={'Ahead': 'Ahead2',
                                'Hour': 'Hour2'},inplace=True)

    TotalTestX=pd.concat([TotalTestX1, TotalTestX2,TotalTestX3],axis=1)
    LocAhead=np.where(TotalTestX.columns=="Ahead")[0][0]
    LocHour=np.where(TotalTestX.columns=="Hour")[0][0]
    TotalTestX.columns=range(0,TotalTestX.shape[1])
    TotalTestX.rename(columns={TotalTestX.columns[LocAhead]:"Ahead",
                               TotalTestX.columns[LocHour]:"Hour"},inplace=True)

    return TotalTrainX, TotalTestX






#######################################################################
## Grouping
#######################################################################
def GroupGen(Decision,TrainX,TestsX):

    if Decision==1:
        NG=1
        TrainGroup=[0]*NG
        TestsGroup=[0]*NG
        TrainGroup[0]=np.arange(0,TrainX.shape[0])
        TestsGroup[0]=np.arange(0,TestsX.shape[0])

    elif Decision==7:
        NG=7
        TrainGroup=[0]*NG
        TestsGroup=[0]*NG
        for a in range(0,7):
            TrainGroup[a]=TrainX[TrainX['Ahead']==TargetAhead[a]].index
            TestsGroup[a]=TestsX[TestsX['Ahead']==TargetAhead[a]].index

    elif Decision==24:
        NG=24
        TrainGroup=[0]*NG
        TestsGroup=[0]*NG
        for a in range(0,24):
            TrainGroup[a]=TrainX[TrainX['Hour']==a].index
            TestsGroup[a]=TestsX[TestsX['Hour']==a].index



    return NG, TrainGroup, TestsGroup




#######################################################################
## Standardization [ General Case ]
#######################################################################
def Standardization ( TrainX, TestsX ):

    # Determine the Unique
    try:
        MegaX=TrainX.append(TestsX,ignore_index=True)
    except:
        MegaX=TrainX.copy()

    STD=np.std(MegaX,axis=0)
    NonZero=np.nonzero(STD)[0]
    Unique=MegaX.iloc[:,NonZero]

    # Normalization
    Normal=(Unique - Unique.mean()) / Unique.std()

    # Set data
    TrainX=Normal.iloc[0:TrainX.shape[0],:]
    TestsX=Normal.iloc[TrainX.shape[0]:,:]

    return TrainX, TestsX



#########################################################################
## Cross Validation Function [ General Case ]
#######################################################################
def CrossVal(Available,TrainX,TrainY):

    CVN=Param['CVN']
    MachineList=Param['MachineList']
    GroupDecision=Param['NGroup']

    ## Cross Validation
    kf = KFold( len(TrainY) ,shuffle=False, n_folds=CVN )

    ## Play the Game
    SavePredict=np.zeros( ( len(MachineList) , len(GroupDecision) , len(TrainY) ) )
    FunRMSE=np.zeros(( len(MachineList) , len(GroupDecision) ))
    for machindex in range(0,len(MachineList)):
        Machine=MachineList[machindex]

        for gt in range(0,len(GroupDecision)):
            GD=GroupDecision[gt]

            c=0
            for train_index, test_index in kf:

                CVTarget = TargetDate - datetime.timedelta(days=CVN) + datetime.timedelta(days=c)
                print CVTarget
                train_index=np.where(Available.available_date!=CVTarget)[0]
                test_index=np.where(Available.available_date==CVTarget)[0]

                c=c+1
                SubTrainX=TrainX.iloc[train_index,:]
                SubTrainY=TrainY.iloc[train_index]
                SubTestsX=TrainX.iloc[test_index,:]
                SubTestsY=TrainY.iloc[test_index]

                # Re-Indexing
                SubTrainX.index=np.arange(0,len(SubTrainX))
                SubTrainY.index=np.arange(0,len(SubTrainY))
                SubTestsX.index=np.arange(0,len(SubTestsX))
                SubTestsY.index=np.arange(0,len(SubTestsY))

                NG,TrainGroup,TestsGroup= GroupGen(GD,SubTrainX, SubTestsX)
                GroupResult=np.zeros(len(SubTestsX))
                for grindex in range(0,1): # NG : Only Whatch the Day+1
                    # Group Assigning
                    TrnX=SubTrainX.iloc[TrainGroup[grindex],:]
                    TrnY=SubTrainY.iloc[TrainGroup[grindex]]
                    TstX=SubTestsX.iloc[TestsGroup[grindex],:]
                    TstY=SubTestsY.iloc[TestsGroup[grindex]]

                    # Outlier Detection
                    #TrnX,TrnY=OutlierDetector(TrnX,TrnY)

                    # Preprocessing
                    TrnX,TstX=Standardization(TrnX, TstX)

                    # Forecast
                    Forecast=Machinery(Machine,TrnX,TrnY,TstX)
                    GroupResult[TestsGroup[grindex]]=Forecast
                    Forecast=Machinery(Machine,TrnX,TrnY,TstX)
                    NormForecast=PostProcessing(Forecast)
                    TempPerformance=PerMeasure(TstY,NormForecast)
                    print Machine, "Main Group:", GD, "Sub Group:", grindex, "CV:", c, TempPerformance

                SavePredict[machindex, gt, test_index]=GroupResult
                NormForecast=PostProcessing(GroupResult)
                TempPerformance=PerMeasure(SubTestsY,NormForecast)
                #print Machine, "Main Group:", GD, "CV:", c, TempPerformance

            SingleOutput=SavePredict[machindex, gt, :]
            NormForecast=PostProcessing(SingleOutput)
            TempPerformance=PerMeasure(TrainY,NormForecast)
            FunRMSE[machindex,gt]=TempPerformance
            #print Machine, "Main Group:", GD, TempPerformance

        SingleOutput=SavePredict[machindex, :, :]
        TempResult=np.mean(SingleOutput,axis=0)
        NormForecast=PostProcessing(TempResult)
        TempPerformance=PerMeasure(TrainY,NormForecast)
        #print Machine, TempPerformance

    TempResult=np.mean(SavePredict,axis=0)
    TempResult=np.mean(TempResult,axis=0)
    NormForecast=PostProcessing(TempResult)
    Forecast=pd.DataFrame(NormForecast,columns=['Forecast'])
    #print FunRMSE, PerMeasure(TrainY,NormForecast)

    return FunRMSE, SavePredict





#######################################################################
## Performance Measure [ Geleral Case ]
#######################################################################
def PerMeasure(X,Y):
    P=np.mean(np.absolute( X - Y ))
    return P


#######################################################################
## Post Processing  [ General Case ]
#######################################################################
def PostProcessing(Prediction):
    # Limitation
    NewPrediction=np.maximum(Prediction,Param['Min'])
    NewPrediction=np.minimum(NewPrediction,Param['Max'])
    return NewPrediction


#######################################################################
## Outlier Detector  [ General Case ]
#######################################################################
def OutlierDetector(TrainX,TrainY):
    Reg = linear_model.LinearRegression()
    loo = LeaveOneOut(len(TrainY))
    MAE=np.array(np.zeros(len(TrainY)))
    id=0
    for train, test in loo:

        SubTrainX=TrainX.iloc[train,:]
        SubTrainY=TrainY.iloc[train]
        SubTestsX=TrainX.iloc[test,:]
        SubTestsY=TrainY.iloc[test]

        # Re-Indexing
        SubTrainX.index=np.arange(0,len(SubTrainX))
        SubTrainY.index=np.arange(0,len(SubTrainY))
        SubTestsX.index=np.arange(0,len(SubTestsX))
        SubTestsY.index=np.arange(0,len(SubTestsY))

        Reg.fit(SubTrainX,SubTrainY)
        TestOutput=Reg.predict(SubTestsX)
        MAE[id]=np.absolute(TestOutput-SubTestsY)
        id=id+1

    Good=MAE.argsort()[:round(len(TrainY)*0.8)]
    TrainX=TrainX.iloc[Good,:]
    TrainY=TrainY.iloc[Good]
    return TrainX, TrainY


########################################################################
## Optimal Alpha Detection
########################################################################
def OptimalReg(TrainX,TrainY):

    CVN=Param['Reg_CVN']
    CVP=Param['Reg_CVP']
    AlphaList=Param['Reg_Reg']

    Performance=np.zeros(len(AlphaList))
    for Alphaindex in range(0,len(AlphaList)):
        Regulator=AlphaList[Alphaindex]

        SubPerformance=np.zeros(CVN)
        for CVindex in range(0,CVN):
            # Set indice for subtrainx and subtestx to perform the CV
            SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=CVP)

            f=Ridge(alpha=Regulator,fit_intercept=True,normalize=False,copy_X=True)
            f.fit(SubTrainX,SubTrainY)

            Output=f.predict(SubTestsX)
            NormOutput=PostProcessing(Output)
            SubPerformance[CVindex]=PerMeasure(NormOutput,SubTestsY.values)

        Performance[Alphaindex]=SubPerformance.mean()

    BestReg=AlphaList[np.argmin(Performance)]
    print "                Best Regulator: ", BestReg, " Performance: ", np.min(Performance)
    return BestReg






########################################################################
## Ridge Regerssion
########################################################################
def RidgeReg(TrainX,TrainY):
    global Param

    BestReg=OptimalReg(TrainX,TrainY)	# Optimal Alpha Detection
    Reg=Ridge(alpha=BestReg,fit_intercept=True,normalize=False,copy_X=True)

    return Reg




########################################################################
## Linear Regerssion
########################################################################
def LinearReg(TrainX,TrainY):
    Reg = linear_model.LinearRegression()
    return Reg






#######################################################################
## Stochastic Gradient Descent
#####################################################################
#def StochasticGradientDescent(TrainX,TrainY):
#	global Param
#	SGD = linear_model.SGDRegressor(loss="squared_loss")
#	return SGD









#######################################################################
## GBM Internal Cross Validation
#####################################################################
def GBMHeldOutScore(CLF,TestsX,TestsY,NTry):

    Score=np.zeros(NTry)
    for i,y_pred in enumerate(CLF.staged_predict(TestsX)):
        Score[i]=PerMeasure(TestsY,y_pred)

    return Score







#######################################################################
## Gradient Boosting Training
#######################################################################
def GradientBoosting(TrainX,TrainY):

#    # Initial Training
#    SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['GBM_CVP'])
#    GBM = GradientBoostingRegressor(n_estimators=Param['GBM_NTry'],
#                                    learning_rate=Param['GBM_LRate'],
#                                    subsample=0.8,verbose=0)
#    GBM.fit(SubTrainX,SubTrainY)
#
#
#	# CV Performance Checking for the number of try
#    CVscore  = GBMHeldOutScore(GBM,SubTestsX,SubTestsY,Param['GBM_NTry'])
#    CVBestIter=np.argmin(CVscore)
#    BestPerformance=np.min(CVscore)
#    print CVBestIter,BestPerformance

    # Retraining
    GBM = GradientBoostingRegressor(n_estimators=Param['GBM_NTry'],
                                    learning_rate=Param['GBM_LRate'],subsample=0.5,verbose=0)

    return GBM







#################################################################################
## KNN
#################################################################################
#def KNearestNeighbors(TrainX , TrainY):
#	global NClass, Param
#	NNeighbors=Param['NNeighbors']
#	KNN=KNeighborsRegressor(n_neighbors=NNeighbors,
#		p=2,
#		algorithm='auto',
#		leaf_size=300,
#		weights='uniform') # ,weights='distance'
#	return KNN







#######################################################################
## XGBoost
#######################################################################
#def XGBoost(TrainX,TrainY):
#	global Param
#	# Model Train
#	xgtrain = xgb.DMatrix(TrainX, label=TrainY)
#	XGB = xgb.train(Param['XGB'], xgtrain, Param['XGB_NR'])
#	return XGB







#################################################################################
## Gaussian Process
#################################################################################
#def GaussianPro(TrainX,TrainY):
#	global Param
#	GP = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#	return GP








#################################################################################
#	## Random Forest
#################################################################################
def RandomForest(TrainX,TrainY):
    global Param

#    SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['RF_CVP'])
#    CandidateLeaf=range(2,3)
#    Score=np.zeros(len(CandidateLeaf))
#    for j in range(0,len(CandidateLeaf)):
#        RF=RandomForestRegressor(n_estimators=50,verbose=0,min_samples_split=CandidateLeaf[j],
# min_samples_leaf=1,bootstrap=True,n_jobs=-1)
#        RF.fit(SubTrainX,SubTrainY)
#
#        # Refitting
#        RF.fit(SubTrainX , SubTrainY)
#        TestOutput=RF.predict(SubTestsX)
#        Prediction=np.maximum(TestOutput,Param['Min'])
#        Score[j]=PerMeasure(SubTestsY,Prediction)
#
#    BestLeafIndex=np.argmin(Score)
#    BestLeaf=CandidateLeaf[BestLeafIndex]
#    BestPerformance=np.min(Score)
#
#    print "BestLeaf: ", BestLeaf, " Best Performance: ", BestPerformance

    NTree=Param['RF_NTree']
    RF=RandomForestRegressor(n_estimators=NTree,verbose=0,min_samples_split=5,
                             min_samples_leaf=1,
                             bootstrap=True,
                             n_jobs=-1)

    return RF



#################################################################################
## Support Vector Machine Classification
#################################################################################
def SupportVectorMachine(TrainX,TrainY):
    # CV Data Spliting
#    SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])
#
#    # Cross validation to select the best C
#    CandidateC=Param['C']
#    Score=np.zeros(len(CandidateC))
#    for i in range(0,len(CandidateC)):
#        Support=SVR(C=CandidateC[i], kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=True,
#			tol=CandidateC[i]/1000.0,
#			cache_size=1000,
#			verbose=0)
#       Support.fit(SubTrainX,SubTrainY)
#       TestOutput=Support.predict(SubTestsX)
#       NewPrediction=PostProcessing(TestOutput)
#       Score[i]=PerMeasure(NewPrediction,SubTestsY)
#
#    BestCIndex=np.argmin(Score)
#    BestC=CandidateC[BestCIndex]
#    BestPerformance=np.min(Score)
#
#    print "BestC: ", BestC, " Best Performance: ", BestPerformance

    # Final Prediction
    Support=SVR(C=Param['C'], kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=True,cache_size=1000,verbose=0)
    #tol=CandidateC[i]/1000.0
    return Support



################################################################################
## Main Machinenary Selection
################################################################################
def Machinery(MachineInfo,TrainX,TrainY,TestsX):

    if MachineInfo=="Reg":
        Attribute=LinearReg(TrainX,TrainY)

    elif MachineInfo=='Ridge':
        Attribute=RidgeReg(TrainX,TrainY)

    elif MachineInfo=='RF':
        Attribute=RandomForest(TrainX,TrainY)

    elif MachineInfo=='GBM':
        Attribute=GradientBoosting(TrainX,TrainY)

    elif MachineInfo=='NN':
        Temp=1

    elif MachineInfo=='SVM':
        Attribute=SupportVectorMachine(TrainX,TrainY)

    elif MachineInfo=='GP':
        Attribute=GaussianPro(TrainX,TrainY)

    elif MachineInfo=='XGB':
        Attribute=XGBoost(TrainX,TrainY)
        xgtest = xgb.DMatrix(TestsX)
        TestOutput = Attribute.predict(xgtest)
        return TestOutput

    elif MachineInfo=='SGD':
        Attribute=StochasticGradientDescent(TrainX,TrainY)

    elif MachineInfo=='KNN':
        Attribute=KNearestNeighbors(TrainX,TrainY)

    # PostProcessing
    Attribute.fit(TrainX,TrainY)
    TestOutput=Attribute.predict(TestsX)

    return TestOutput




######################################################################
## Ensemble Classification
#####################################################################
def Ensemble(FunRMSE, SavePredict, TrainY):
    EnsemOption=Param['Ensem']
    ReshapeFunRMSE=np.reshape(FunRMSE,FunRMSE.size)
    ReshapeSavePredict=np.reshape(SavePredict,(FunRMSE.size,SavePredict.shape[2]))

    if EnsemOption=='Best':
        # Weight
        Weight=np.array([1])

        # Result Making
        TempResult=ReshapeSavePredict[np.argmin(ReshapeFunRMSE)]

        # Best Index
        BestGroup=np.where(FunRMSE==np.min(FunRMSE))[0][0]
        BestGroup=np.array(BestGroup)
        BestMachine=np.where(FunRMSE==np.min(FunRMSE))[1][0]
        BestMachine=np.array(BestMachine)

    elif EnsemOption=='Simple':
        # Weight
        Weight=np.ones(FunRMSE.size)*1.0/FunRMSE.size
        Weight=Weight[np.newaxis].transpose()

        # Result Making
        TempResult=np.mean(ReshapeSavePredict,axis=0)

        # Best Index
        BestGroup=range(0,FunRMSE.shape[1])
        BestMachine=range(0,FunRMSE.shape[0])

    elif EnsemOption=='Weight':
        # Weight
        InverseWeight=1.0 / ReshapeFunRMSE
        Weight=InverseWeight / sum(InverseWeight)
        Weight=Weight[np.newaxis].transpose()

        # Result Making
        TempResult= Weight * ReshapeSavePredict
        TempResult=np.sum(TempResult,axis=0)

        # Best Index
        BestGroup=range(0,FunRMSE.shape[1])
        BestMachine=range(0,FunRMSE.shape[0])

    # Performance Measure
    NormForecast=PostProcessing(TempResult)
    PrintOutRMSE=PerMeasure(TrainY,NormForecast)
    print FunRMSE, PrintOutRMSE

    return NormForecast, Weight, BestGroup, BestMachine



#################################################################################
## Quick Ensemble
##################################################################################
def QuickEnsemble(Param):
    GameSize=len(Param['NGroup']) * len(Param['MachineList'])
    Weight=np.ones(GameSize)*1.0/GameSize
    Weight=Weight[np.newaxis].transpose()
    BestGroup=range(0,len(Param['NGroup']))
    BestMachine=range(0,len(Param['MachineList']))
    return Weight,BestGroup,BestMachine


#################################################################################
## Second Wave
##################################################################################
def SecondWave(MeanPredict,WeightedOutput,TrainX,TestsX):

    Forecast=pd.DataFrame(MeanPredict,columns=['Forecast'])
    WeightedOutput=pd.DataFrame(WeightedOutput,columns=['Forecast'])

    NewTrainX=pd.concat([TrainX,Forecast],axis=1)
    NewTestsX=pd.concat([TestsX,WeightedOutput],axis=1)

    return NewTrainX, NewTestsX




####################################################################################
## Final Training
##################################################################################
def FinalTraining(TrainX,TrainY,TestsX,Weight,BestGroup,BestMachine):
    MachineList=Param['MachineList']
    GroupDecision=Param['NGroup']

    SavePredict=np.zeros( ( len(BestMachine) , len(BestGroup) , len(TestsX) ) )
    for machindex in range(0,len(BestMachine)):
        Machine=MachineList[BestMachine[machindex]]

        for gt in range(0,len(BestGroup)):
            GD=GroupDecision[BestGroup[gt]]
            NG,TrainGroup,TestsGroup= GroupGen(GD,TrainX, TestsX)
            #GroupResult=np.zeros( len(TestsX))
            for grindex in range(0,NG):

                # Group Assigning
                TrnX=TrainX.iloc[TrainGroup[grindex],:]
                TrnY=TrainY.iloc[TrainGroup[grindex]]
                TstX=TestsX.iloc[TestsGroup[grindex],:]

                # Outlier Detection
                #TrnX,TrnY=OutlierDetector(TrnX,TrnY)

                # Preprocessing
                TrnX,TstX=Standardization(TrnX, TstX)

                # Forecast
                Forecast=Machinery(Machine,TrnX,TrnY,TstX)
                SavePredict[machindex,gt,TestsGroup[grindex]]=Forecast


    # Weighted Output Generation
    ReshapeSavePredict=np.reshape(SavePredict,(SavePredict.shape[0]*SavePredict.shape[1],SavePredict.shape[2]))
    TempResult= Weight * ReshapeSavePredict
    WeightedOutput=np.sum(TempResult,axis=0)
    WeightedOutput=PostProcessing(WeightedOutput)

    return WeightedOutput



def LoadForecasting(WeatherData,PTAverageGen):

    WeatherDemand=pd.merge(WeatherData ,PTAverageGen,how="left",on="prediction_date")

    ##
    DemandTestX=WeatherDemand[(WeatherDemand.available_date==TargetDate) | (pd.isnull(WeatherDemand["Demand"]))]
    NextGenTest = DemandTestX[["Demand","available_date","prediction_date"]]
    DemandTestX["Hour"]=pd.DatetimeIndex(DemandTestX["prediction_date"]).hour
    DemandTestX=DemandTestX.drop(["Year","Month","Day","Demand","available_date","available_date2","prediction_date"],1)
    DemandTestX.index=range(0,DemandTestX.shape[0])

    DemandTrainX=WeatherDemand[(WeatherDemand.available_date!=TargetDate) & (~pd.isnull(WeatherDemand["Demand"]))]
    NextGenTrain = DemandTrainX[["Demand","available_date","prediction_date"]]
    DemandTrainY=DemandTrainX["Demand"]
    DemandTrainY.index=range(0,DemandTrainY.shape[0])

    DemandTrainX=DemandTrainX.drop(["Year","Month","Day","Demand","available_date","available_date2","prediction_date"],1)
    DemandTrainX.index=range(0,DemandTrainX.shape[0])

    ##
    #FunRMSE, LoadPredict=CrossVal(DemandTrainX,DemandTrainY)

    #TrainOutput,Weight,BestGroup,BestMachine=Ensemble(FunRMSE,LoadPredict,DemandTrainY)

    Weight, BestGroup, BestMachine = QuickEnsemble(Param)
    LoadForecast=FinalTraining(DemandTrainX,DemandTrainY,DemandTestX,Weight,BestGroup,BestMachine)
    NextGenTest["Demand"]=LoadForecast

    DemandSupport=pd.concat([NextGenTrain,NextGenTest],axis=0)

    #NewTrainX, NewTestsX = SecondWave(TrainOutput,LoadForecast,DemandTrainX,DemandTestX)
    #
    #FunRMSE2, LoadPredict2 = CrossVal(NewTrainX,DemandTrainX)
    #
    #TrainOutput2,Weight2,BestGroup2,BestMachine2=Ensemble(FunRMSE2,LoadPredict2,DemandTrainY)
    #
    #LoadForecast2=FinalTraining(NewTrainX,DemandTrainX,NewTestsX,Weight2, BestGroup2, BestMachine2)

    return DemandSupport



########################################################################
# Printing
########################################################################
def PostPrinting(TargetDateTime, SubmissionDate, StartAhead, Prediction):
    # Prepare to print it out
    StartPoint = (StartAhead-1)*24
    SelectForecast = Prediction[ StartPoint : (StartPoint + 120)]

    Template = pd.read_csv("/Users/dueheelee/Documents/ComPlatt/Template/Template.csv")

    Available = datetime.datetime.strptime(SubmissionDate, "%Y-%m-%d")
    AvailableString = Available.strftime('%d-%m-%Y')
    Template["availabledate"] = AvailableString

    PredictionDate=['a']*5
    for i in range(0,5):
        TempDate = Available + datetime.timedelta(days=(i+1))
        PredictionDate[i]=TempDate.strftime('%d-%m-%Y')

    Template["predictiondate"] = np.append(np.array(AvailableString),np.repeat(PredictionDate,24)[:-1])
    Template["value"] = SelectForecast

    ##
    Template.to_csv(SubmissionFolder+ForecastorID+"_"+SubmissionDate+".csv",index=False,heater=True)
    return Template











# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Parameters
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
SubmissionDate = "2016-03-27"
TargetDate=pd.to_datetime(SubmissionDate + " 06:00:00")
StartAhead=1
Ahead=np.repeat(range(1,8),24)
TimeIndex=["available_date","prediction_date"]
TargetAhead=range(1,8)
SubmissionFolder="/Users/dueheelee/Documents/ComPlatt/Submission/"
ForecastorID = "4c17235d2f125cb17cb2d782fce8d201"


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         User Parameters Calling
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Param=ParameterSetting()


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Old Information
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
OldPriceLoc = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Price/"
OldWeatherLoc = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Weather/"
OldWeatherPrice=OldWeatherPriceRead(OldPriceLoc, "RealMarketPriceDataPT.csv", OldWeatherLoc, "historical_weather.csv")



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Price Information
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
PriceFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Historical_Price/"
RawPrice = PriceReader(PriceFileLoc,"NewPrice.csv")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
PTGenerationLoc = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Generation/"
# PTAverageGen = PTGenerationReader(PTGenerationLoc,"RealDataPT.csv")
PTAverageGen = ESGenerationReader(PTGenerationLoc,"RealDataES.csv")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Weahter Data Collection
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Forecast_Weather/"
WeatherData = SingleWeather(WeatherFileLoc)








# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Load Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
BeforePrice = RawPrice.copy()
del RawPrice["DayOnly"]
BeforePrice.rename(columns={'Price':'AheadPrice',
                            'prediction_date':'available_date2'}, inplace=True)

## Daily Average Price
AveragePrice=pd.pivot_table(BeforePrice,values="AheadPrice",index=["DayOnly"],aggfunc=np.mean)
AveragePrice=pd.DataFrame(AveragePrice).reset_index()
AveragePrice.rename(columns={'AheadPrice': 'AveragePrice'}, inplace=True)
AveragePrice["available_date2"]=pd.to_datetime(AveragePrice["DayOnly"]+" 23:00:00")

#if StartAhead==2:
#    WeatherData["available_date2"]=pd.DatetimeIndex(WeatherData["available_date2"]).date + datetime.timedelta(days=1)

DataUpgrade = pd.merge(WeatherData,BeforePrice[["available_date2","AheadPrice"]],how="left",on="available_date2")
DataUpgrade2 = pd.merge(DataUpgrade,AveragePrice[["available_date2","AveragePrice"]],how="left",on="available_date2")






# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherPrice=pd.merge(DataUpgrade2 ,RawPrice,how="left",on="prediction_date")



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Load Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
DemandSupport = LoadForecasting(WeatherData,PTAverageGen)
WeatherPrice=pd.merge(WeatherPrice ,DemandSupport,how="left",on=["prediction_date","available_date"])


##
TotalTestX=WeatherPrice[WeatherPrice.available_date==TargetDate]
TotalTestX["Hour"]=pd.DatetimeIndex(TotalTestX["prediction_date"]).hour
TotalTestX=TotalTestX.drop(["Year","Month","Day","Price","available_date","available_date2","prediction_date"],1)
TotalTestX.index=range(0,TotalTestX.shape[0])

TotalTrainX=WeatherPrice[(WeatherPrice.available_date!=TargetDate) & (~pd.isnull(WeatherPrice["Price"]))]
TotalTrainY=TotalTrainX["Price"]
Available=TotalTrainX[["available_date","Ahead"]]
TotalTrainY.index=range(0,TotalTrainY.shape[0])

TotalTrainX=TotalTrainX.drop(["Year","Month","Day","Price","available_date","available_date2","prediction_date"],1)
TotalTrainX.index=range(0,TotalTrainX.shape[0])

## Extention Process
AheadLoc=np.where(TotalTrainX.columns=="Ahead")[0][0]
HourLoc=np.where(TotalTrainX.columns=="Hour")[0][0]
DemandLoc=np.where(TotalTrainX.columns=="Demand")[0][0]
Selection=[ 48,   0, 178, 156, 166, 172,  44, 176, 159,  73, 136,  60,  11,
           162,   3, 168,  86, 126,  30, 181, 116, 128, 146, 119,  43,  72,
           6, 164,  78, 141,   1, 182, 138,  62,  36, 113, 158,  28,  46,
           142, 102,  22,  76,  38,  58, 151, 131, 123,  52, 124, 110, 101,
           51, 133,  53,  42, 103, 161,   9,  13, 173,  20, 148,  81, 150,
           82,  68,  31, 183, 122,  16,  96,  91, 130, 144,   4, 153,  26, 12,  33]
Selection.append(AheadLoc)
Selection.append(HourLoc)
Selection.append(DemandLoc)
Selection=np.unique(np.array(Selection))
#TotalTrainX, TotalTestX = TrainingExtention(Selection, TotalTrainX, TotalTestX)
#TotalTrainX=TotalTrainX.iloc[:,Selection]
#TotalTestX=TotalTestX.iloc[:,Selection]





Answer2=np.array([13.69,10.69,14.19,14.19,14.19,10.43,8,14.82,15.69,18.80,15.05,14.19,14.82,13,9.20,8,
                  11.16,15,20.94,30,30,28.01,21.19])


#Answer2=np.array([30,24.2,21.11,20,20,20.1,20.1,21.69,22.49,24,23.03,20,19.06,
#                 18.09,16.28,14,14.38,14.38,17.63,22.12,21.69,23,20,18.6])
#
#Answer=np.concatenate([Answer1,Answer2])

Param['MachineList']=['Reg']
Param['GBM_NTry']=1000
Param['GBM_LRate']=0.01
Param['GBM_CVP']=0.1
Param['NGroup']=[1]

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Cross Validation Ticket Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
FunRMSE, SavePredict=CrossVal(Available,TotalTrainX,TotalTrainY)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Ensemble Function and Weights
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# TrainOutput,Weight,BestGroup,BestMachine=Ensemble(FunRMSE,SavePredict,TotalTrainY)




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Final Training 1
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Weight, BestGroup, BestMachine = QuickEnsemble(Param)
SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine)
MAE=np.absolute(SaveForecast[:23]-Answer2).mean()
print MAE



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Preparing Second Race
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# NewTrainX, NewTestsX = SecondWave(TrainOutput,SaveForecast,TotalTrainX,TotalTestX)




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          The Second Cross Validation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#FunRMSE2, SavePredict2 = CrossVal(NewTrainX,TotalTrainY)





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          The Second Ensemble
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#TrainOutput2,Weight2,BestGroup2,BestMachine2=Ensemble(FunRMSE2,SavePredict2,TotalTrainY)

#TotalTrainY=pd.DataFrame(TotalTrainY)
#TotalTrainY["Predict"]=SavePredict2[0,0,:]
#TotalTrainY.to_csv('Compare.csv')





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          The Second Final Training
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#Weight2=Weight
#BestGroup2=BestGroup
#BestMachine2=BestMachine
#SaveForecast2=FinalTraining(NewTrainX,TotalTrainY,NewTestsX,Weight2, BestGroup2, BestMachine2)
#MAE=np.absolute(SaveForecast2[:24]-Answer).mean()
#print MAE

#SaveForecast2=pd.DataFrame(SaveForecast2)
#SaveForecast2.to_csv('Predictions.csv')




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Printing
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Result = PostPrinting(TargetDate, SubmissionDate, StartAhead, SaveForecast)
#Result.head()


