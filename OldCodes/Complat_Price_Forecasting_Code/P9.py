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
pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 50000)
pd.set_option('display.height', 50000)


#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def ParameterSetting():
    # Cross Validation Parameters
    Param={'CVN': 5, 'CVP':0.1, 'CVOptExternal':"HoldOut"}

    # Machine List
    Param['MachineList']=['RF']   # LogReg KNN GBM 'LogReg', ,'Ridge' "Reg"

    # Group Number
    Param['NGroup']=[1]

    # Ensemble Option
    Param['Ensem']="Simple"

    # Internal Cross Validation Option
    Param['CVOptInternal']="HoldOut"

    # GBM
    Param['GBM_NTry']=200
    Param['GBM_LRate']=0.02
    Param['GBM_CVP']=0.4

    # Random Forest
    Param['RF_NTree']=500  # 200
    Param['RF_CVP']=0.1

    # SVM
    Param['SVM_CVP']=0.5
    Param['C']=[1] #[15,20,25,30] # [3,5,7,8,9,10]
    Param['Gamma']=0.01

	# NuSVM
    Param['SVM_CVP']=0.4
    Param['Nu']=[0.0001, 0.0002, 0.0003, 0.0005, 0.001]

    # KNN
#	Param['NNeighbors']=3000
#
    # Ridge Regression
    Param['Reg_Reg']=[1,5,10,15,20,30,40,50,100,200,300,500,1000]
    Param['Reg_CVN']=5
    Param['Reg_CVP']=0.4

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
## Price Reader
#######################################################################
def PriceReader(PriceFileLoc,PriceFileName):
    RawPrice  = pd.read_csv(PriceFileLoc + PriceFileName)
    RawPrice["prediction_date"]=pd.to_datetime(RawPrice["prediction_date"])
    return RawPrice




#######################################################################
## Old Weather Reader
#######################################################################
def OldWeatherRead(OldWeatherLoc, OldWeatherFile):

    OldWeather = pd.read_csv(OldWeatherLoc + OldWeatherFile)
    del OldWeather["precipitation"]

    Single = OldWeather[ OldWeather.point==1]
    del Single["point"]
    for t in range(2,19):
        Temp2 = OldWeather[ OldWeather.point==t]
        del Temp2["point"]
        Single = pd.merge(Single,Temp2,on=["prediction_date","Year","Month","Day","Hour"],suffixes=('',str(t)))

    Single["Ahead"] = 1

    Single["prediction_date"]=pd.to_datetime(Single["prediction_date"])
    Single["available_date"]=Single["prediction_date"] - datetime.timedelta(days=1)
    Single["available_date"]= pd.DatetimeIndex(Single["available_date"]).date
    Single["available_date"]=pd.to_datetime(Single["available_date"])

    Single["WeekDay"]=pd.DatetimeIndex(Single["prediction_date"]).weekday

    Single=Single[Single.Year>2014]
    Single.index=range(0,Single.shape[0])

    return Single




#######################################################################
## Weather Data Reading
#######################################################################
def SingleWeather(WeatherFileLoc):
    FileList=os.listdir(WeatherFileLoc)
    FileList.sort()

    # os.remove() will remove a file.

    ##
    for f in range(0,len(FileList)):
        ReadData  = pd.read_csv(WeatherFileLoc + FileList[f])
        del ReadData["precipitation"]

        Single = ReadData[ ReadData.point==1]
        del Single["point"]
        for t in range(2,19):
            Temp2 = ReadData[ ReadData.point==t]
            del Temp2["point"]
            Single = pd.merge(Single,Temp2,on=TimeIndex,suffixes=('',str(t)))

        Single["Ahead"] = Ahead

        # Select only less than 5
        #Single=Single[Single["Ahead"]<6]

        if f==0:
            Data = Single
        else:
            Data = pd.concat([Data,Single])

    Data.index=range(0,Data.shape[0])
    Data["prediction_date"]=pd.to_datetime(Data["prediction_date"])
    Data["available_date"]= pd.DatetimeIndex(Data["available_date"]).date
    Data["available_date"]=pd.to_datetime(Data["available_date"])


    Data["WeekDay"]=pd.DatetimeIndex(Data["prediction_date"]).weekday
    Data["Year"]=pd.DatetimeIndex(Data["prediction_date"]).year
    Data["Month"]=pd.DatetimeIndex(Data["prediction_date"]).month
    Data["Day"]=pd.DatetimeIndex(Data["prediction_date"]).day
    Data["Hour"]=pd.DatetimeIndex(Data["prediction_date"]).hour

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
    #PTGen['Gen'] = PTGen.iloc[:,range(6,20)].sum(axis=1)

    TimeIndexPT=["Date","Year","Month","Day","Hour"]
    PTAverageGen = pd.pivot_table(PTGen,values="real_demand",index=TimeIndexPT,aggfunc=np.mean)
    PTAverageGen=pd.DataFrame(PTAverageGen).reset_index()
    PTAverageGen.rename(columns={'real_demand': 'Demand'}, inplace=True)
    PTAverageGen.rename(columns={'Date': 'prediction_date'},inplace=True)
    return PTAverageGen




########################################################################
# Training Extention
########################################################################
def TrainingExtention(TotalTrainX, TotalTestX):
    TempNumber1=TotalTestX.shape[0]
    TempNumber0=TempNumber1-1


    TotalTrainX2=TotalTrainX #.iloc[:,Selection]
    TotalTestX2=TotalTestX #.iloc[:,Selection]

    TotalTrainX1=pd.concat([TotalTrainX2.iloc[0:1,:], TotalTrainX2.iloc[:-1,:]],axis=0)
    TotalTrainX1.index=range(0,TotalTrainX1.shape[0])
    TotalTrainX1.rename(columns={'Ahead': 'Ahead2',
                                 'Hour': 'Hour2'},inplace=True)
    TotalTrainX3=pd.concat([TotalTrainX2.iloc[1:,:], TotalTrainX2.iloc[TempNumber0:TempNumber1,:]],axis=0)
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
    TotalTestX3=pd.concat([TotalTestX2.iloc[1:,:], TotalTestX2.iloc[TempNumber0:TempNumber1,:]],axis=0)
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

    elif Decision==2:
        NG=2
        TrainGroup=[0]*NG
        TestsGroup=[0]*NG
        for a in range(0,2):
            TrainGroup[a]=TrainX[TrainX['Ahead']==(a+1)].index
            TestsGroup[a]=TestsX[TestsX['Ahead']!=(a+1)].index

    elif Decision==5:
        NG=5
        TrainGroup=[0]*NG
        TestsGroup=[0]*NG
        for a in range(0,5):
            TrainGroup[a]=TrainX[TrainX['Ahead']==TargetAhead[a]].index
            TestsGroup[a]=TestsX[TestsX['Ahead']==TargetAhead[a]].index

    elif Decision==7:
        NG=7
        TrainGroup=[0]*NG
        TestsGroup=[0]*NG
        for a in range(0,7):
            TrainGroup[a]=TrainX[TrainX['WeekDay']==a].index
            TestsGroup[a]=TestsX[TestsX['WeekDay']==a].index

    elif Decision==8:
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
def CrossVal(AvailableDate,TrainX,TrainY,Param2):

    CVN=Param2['CVN']
    MachineList=Param2['MachineList']
    GroupDecision=Param2['NGroup']

    ## Cross Validation
    kf = KFold( len(TrainY) ,shuffle=False, n_folds=CVN )

    ## Play the Game
    SavePredict=np.zeros( ( len(MachineList) , len(GroupDecision) , len(TrainY) ) )
    FunRMSE=np.zeros(( len(MachineList) , len(GroupDecision) ))
    for machindex in range(0,len(MachineList)):
        Machine=MachineList[machindex]

        for gt in range(0,len(GroupDecision)):
            GD=GroupDecision[gt]

            cb=0
            for train_index, test_index in kf:

                CVTarget = TargetDate - datetime.timedelta(days=(CVN+GD+5)) + datetime.timedelta(days=cb)
                print CVTarget
                train_index=np.where(AvailableDate.prediction_date!=CVTarget)[0]
                test_index=np.where(AvailableDate.prediction_date==CVTarget)[0]

                cb=cb+1
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
                for grindex in range(0,NG): # NG : Only Whatch the Day+1 min(5,NG)
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
                    NormForecast=PostProcessing(Forecast)
                    TempPerformance=PerMeasure(TstY,NormForecast)
                    print Machine, "Main Group:", GD, "Sub Group:", grindex, "CV:", cb, TempPerformance

                SavePredict[machindex, gt, test_index]=GroupResult
                NormForecast=PostProcessing(GroupResult)
                TempPerformance=PerMeasure(SubTestsY,NormForecast)
                #print Machine, "Main Group:", GD, "CV:", cb, TempPerformance

            SingleOutput=SavePredict[machindex, gt, :]
            Available=np.nonzero(SingleOutput)[0]
            SingleOutput=SingleOutput[Available]
            NormForecast=PostProcessing(SingleOutput)
            TempPerformance=PerMeasure(TrainY[Available],NormForecast)
            FunRMSE[machindex,gt]=TempPerformance
            print Machine, "Main Group:", GD, TempPerformance

        SingleOutput=SavePredict[machindex, :, :]
        TempResult=np.mean(SingleOutput,axis=0)
        Available=np.nonzero(TempResult)[0]
        TempResult=TempResult[Available]

        NormForecast=PostProcessing(TempResult)
        TempPerformance=PerMeasure(TrainY[Available],NormForecast)
        #print Machine, TempPerformance

    TempResult=np.mean(SavePredict,axis=0)
    TempResult=np.mean(TempResult,axis=0)
    Available=np.nonzero(TempResult)[0]
    TempResult=TempResult[Available]
    NormForecast=PostProcessing(TempResult)
    Forecast=pd.DataFrame(NormForecast,columns=['Forecast'])
    print FunRMSE, PerMeasure(TrainY[Available],NormForecast)

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
    Reg.fit(TrainX , TrainY)
    TestOutput=Reg.predict(TrainX)
    MAE=np.absolute(TestOutput-TrainY)
    MAE=np.array(MAE)
    Good=MAE.argsort()[:round(len(TrainY)*0.8)]
    TrainX=TrainX.iloc[Good,:]
    TrainY=TrainY.iloc[Good]
    return TrainX, TrainY


#def OutlierDetector(TrainX,TrainY):
#    Reg = linear_model.LinearRegression()
#    loo = LeaveOneOut(len(TrainY))
#    MAE=np.array(np.zeros(len(TrainY)))
#    id=0
#    for train, test in loo:
#
#        SubTrainX=TrainX.iloc[train,:]
#        SubTrainY=TrainY.iloc[train]
#        SubTestsX=TrainX.iloc[test,:]
#        SubTestsY=TrainY.iloc[test]
#
#        # Re-Indexing
#        SubTrainX.index=np.arange(0,len(SubTrainX))
#        SubTrainY.index=np.arange(0,len(SubTrainY))
#        SubTestsX.index=np.arange(0,len(SubTestsX))
#        SubTestsY.index=np.arange(0,len(SubTestsY))
#
#        Reg.fit(SubTrainX,SubTrainY)
#        TestOutput=Reg.predict(SubTestsX)
#        MAE[id]=np.absolute(TestOutput-SubTestsY)
#        id=id+1
#
#    Good=MAE.argsort()[:round(len(TrainY)*0.9)]
#    TrainX=TrainX.iloc[Good,:]
#    TrainY=TrainY.iloc[Good]
#    return TrainX, TrainY


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
    RF=RandomForestRegressor(n_estimators=NTree,verbose=0,min_samples_split=2,criterion='mse',
                             bootstrap=True,max_features="auto",
                             n_jobs=-1)

    # min_samples_leaf=1,
    #
    # min_samples_split
    #

    return RF



#################################################################################
## Support Vector Machine Classification
#################################################################################
def SupportVectorMachine(TrainX,TrainY):
    # CV Data Spliting
    SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])

    # Cross validation to select the best C
    CandidateC=Param['C']
    Score=np.zeros(len(CandidateC))
    for i in range(0,len(CandidateC)):
        Support=SVR(C=CandidateC[i], kernel='rbf', degree=3,gamma=Param['Gamma'],shrinking=True,tol=CandidateC[i]/1000.0,cache_size=1000,verbose=0)
        Support.fit(SubTrainX,SubTrainY)
        TestOutput=Support.predict(SubTestsX)
        NewPrediction=PostProcessing(TestOutput)
        Score[i]=PerMeasure(NewPrediction,SubTestsY)

    BestCIndex=np.argmin(Score)
    BestC=CandidateC[BestCIndex]
    BestPerformance=np.min(Score)

    print "BestC: ", BestC, " Best Performance: ", BestPerformance

    # Final Prediction

    Support=SVR(C=BestC, kernel='rbf', degree=3, epsilon=0.1,  gamma=Param['Gamma'], shrinking=True,cache_size=1000,verbose=0)
    #tol=CandidateC[i]/1000.0
    return Support

# Param['Gamma']

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

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##    Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def WeatherPriceCombine(TotalWeather,Price):

    WeatherPrice=pd.merge(TotalWeather, Price,how="left",on=["prediction_date","Year","Month","Day","Hour"])
    WeatherPrice["DayOnly"]= pd.DatetimeIndex(WeatherPrice["available_date"]).date
    WeatherPrice["DayOnly"]= pd.to_datetime(WeatherPrice["DayOnly"])

    BeforePrice = Price.copy()
    BeforePrice["DayOnly"]= pd.DatetimeIndex(BeforePrice["prediction_date"]).date
    BeforePrice["DayOnly"]=pd.to_datetime(BeforePrice["DayOnly"])
    BeforePrice=BeforePrice[BeforePrice.Hour==23]
    BeforePrice.rename(columns={'Price':'AheadPrice'}, inplace=True)

    ## Daily Average Price
    AveragePrice=pd.pivot_table(BeforePrice,values="AheadPrice",index=["DayOnly"],aggfunc=np.mean)
    AveragePrice=pd.DataFrame(AveragePrice).reset_index()
    AveragePrice.rename(columns={'AheadPrice': 'AveragePrice'}, inplace=True)

    DataUpgrade = pd.merge(WeatherPrice,BeforePrice[["DayOnly","AheadPrice"]],how="left",on="DayOnly")
    DataUpgrade2 = pd.merge(DataUpgrade,AveragePrice[["DayOnly","AveragePrice"]],how="left",on="DayOnly")

    DataUpgrade2["DayOnly"]
    return DataUpgrade2



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##         Load Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def LoadForecasting(WeatherData,PTAverageGen,DemandPT,Param2):

    WeatherDemand=pd.merge(WeatherData,PTAverageGen,how="left",on=["prediction_date","Year","Month","Day","Hour"])
    WeatherDemand=pd.merge(WeatherDemand,DemandPT,how="left",on=["prediction_date","Year","Month","Day","Hour"])

    ##
    DemandTestX=WeatherDemand[(WeatherDemand.available_date==TargetDate) | (pd.isnull(WeatherDemand["Demand"]))]

    NextGenTest = DemandTestX[["Demand","available_date","prediction_date"]]
    DemandTestX=DemandTestX.drop(["Demand","available_date","prediction_date"],1)
    DemandTestX.index=range(0,DemandTestX.shape[0])

    DemandTrainX=WeatherDemand[(WeatherDemand.available_date!=TargetDate) & (~pd.isnull(WeatherDemand["Demand"]))]
    NextGenTrain = DemandTrainX[["Demand","available_date","prediction_date"]]

    DemandTrainX=DemandTrainX[~(DemandTrainX.isnull().any(axis=1))]
    AvailableDate=DemandTrainX[["prediction_date","Ahead"]]

    DemandTrainY=DemandTrainX["Demand"]
    DemandTrainY.index=range(0,DemandTrainY.shape[0])

    DemandTrainX=DemandTrainX.drop(["Demand","available_date","prediction_date"],1)
    DemandTrainX.index=range(0,DemandTrainX.shape[0])

    #DemandTrainX, DemandTestX = TrainingExtention(DemandTrainX, DemandTestX)


    Param2['MachineList']=["RF"]
    Param2['GBM_NTry']=500
    Param2['RF_NTree']=100
    Param2['NGroup']=[1]



    FunRMSE, LoadPredict=CrossVal(AvailableDate,DemandTrainX,DemandTrainY,Param2)

    #TrainOutput,Weight,BestGroup,BestMachine=Ensemble(FunRMSE,LoadPredict,DemandTrainY)

    Weight, BestGroup, BestMachine = QuickEnsemble(Param2)
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

    Seed=np.array(range(0,24))
    Seed=np.tile(Seed,5)
    Seed=np.append([22,23],Seed)
    Seed=Seed[:-2]


    Template=pd.DataFrame(Seed,columns=["hour"])
    Template["forecaster"]=ForecastorID


    Available = datetime.datetime.strptime(SubmissionDate, "%Y-%m-%d")
    AvailableString = Available.strftime('%d/%m/%Y')
    Template["availabledate"] = AvailableString
    Template["predictiondate"] = AvailableString

    PredictionDate=['a']*5
    for i in range(0,5):
        TempDate = Available + datetime.timedelta(days=(i+1))
        PredictionDate[i]=TempDate.strftime('%d/%m/%Y')

    Temp = np.repeat(PredictionDate,24)

    Template.loc[2:,"predictiondate"]= Temp[:-2]
    Template["value"] = SelectForecast

    Template=Template[["forecaster","availabledate","predictiondate","hour","value"]]

    ##
    Template.to_csv(SubmissionFolder+ForecastorID+"_"+SubmissionDate+".csv",index=False,heater=True)
    return Template















# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Parameters
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
SubmissionDate = "2016-04-03"
TargetDate=pd.to_datetime(SubmissionDate)
StartAhead=1
Ahead=np.repeat(range(1,8),24)
TimeIndex=["available_date","prediction_date"]
TargetAhead=range(1,8)
SubmissionFolder="/Users/dueheelee/Documents/ComPlatt/Submission/"
ForecastorID = "4c17235d2f125cb17cb2d782fce8d201"


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         User Parameters Calling
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Param=ParameterSetting()


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Price Information
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
PriceFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Historical_Price/"
Price = PriceReader(PriceFileLoc,"NewPrice.csv")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Old Weather Information
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
OldWeatherLoc = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Weather/"
OldWeather=OldWeatherRead(OldWeatherLoc, "historical_weather.csv")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############      New Weahter Data Collection
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Forecast_Weather/"
NewWeather = SingleWeather(WeatherFileLoc)
TotalWeather=pd.concat([OldWeather,NewWeather])


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
PTGenerationLoc = "/Users/dueheelee/Documents/ComPlatt/Data/Forecast_PTDemand/"
# PTAverageGen = PTGenerationReader(PTGenerationLoc,"RealDataPT.csv")
PTAverageGen = ESGenerationReader(PTGenerationLoc,"ESDemand.csv")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Energy
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Energy = pd.read_csv(PTGenerationLoc + "PTHourlyDemand.csv")  # PTHourlyDemand.csv  PDEnergy.csv
Energy["prediction_date"]=pd.to_datetime(Energy["prediction_date"])



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherPrice = WeatherPriceCombine(TotalWeather,Price)



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Load Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
DemandSupport = LoadForecasting(TotalWeather,PTAverageGen,Energy,Param)
WeatherPrice=pd.merge(WeatherPrice ,DemandSupport,how="left",on=["prediction_date","available_date"])
WeatherPrice=pd.merge(WeatherPrice ,Energy[["prediction_date","DemandPT"]],how="left",on=["prediction_date"])



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Training and Testing Data Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Threshold=Price.loc[Price.shape[0]-1,"prediction_date"]
TotalTestX=WeatherPrice[WeatherPrice.prediction_date > Threshold]
TotalTestX=TotalTestX.drop(["Price","DayOnly","available_date","prediction_date"],1)
TotalTestX.index=range(0,TotalTestX.shape[0])

# & (~pd.isnull(WeatherPrice["Price"]))

TotalTrainX=WeatherPrice[(WeatherPrice.prediction_date <= Threshold) ]
TotalTrainX=TotalTrainX[~(TotalTrainX.isnull().any(axis=1))]
TotalTrainY=TotalTrainX["Price"]
AvailableDate=TotalTrainX[["prediction_date","Ahead"]]
TotalTrainY.index=range(0,TotalTrainY.shape[0])

TotalTrainX=TotalTrainX.drop(["Price","DayOnly","available_date","prediction_date"],1)
TotalTrainX.index=range(0,TotalTrainX.shape[0])



## Extention Process
#TotalTrainX, TotalTestX = TrainingExtention(TotalTrainX, TotalTestX)


Answer=np.array([19.59,19.15,18.75,18.72,20.28,24.69,24.75,24.69,25.96,29.06,29.23,29.55,
                 29.01,29,29.69,27.59,29.25,29.69,29.69,30.44,34.75,30.30,25.96])


Param['MachineList']=["RF"]
Param['GBM_NTry']=500
Param['RF_NTree']=100
Param['NGroup']=[1]



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Cross Validation Ticket Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
FunRMSE, SavePredict=CrossVal(AvailableDate,TotalTrainX,TotalTrainY,Param)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Ensemble Function and Weights
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# TrainOutput,Weight,BestGroup,BestMachine=Ensemble(FunRMSE,SavePredict,TotalTrainY)




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Final Training 1
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Weight, BestGroup, BestMachine = QuickEnsemble(Param)
SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine)
TotalTestX["Forecast"]=SaveForecast

FinalValue=np.zeros(120)
for i in range(0,5):
    A=TargetDate + datetime.timedelta(days=(i+1))
    Sub=TotalTestX[TotalTestX.prediction_date==A]
    Sub=Sub[["prediction_date","Forecast"]]
    AverageForecast=pd.pivot_table(Sub,values="Forecast",index=["prediction_date"],aggfunc=np.mean)
    FinalValue[i*24:(i+1)*24]=AverageForecast

MAE=np.absolute(FinalValue[:23]-Answer).mean()
print FunRMSE.mean(), '  /  ', MAE, '  /  ' , (MAE+FunRMSE.mean())/2




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

#Result = PostPrinting(TargetDate, SubmissionDate, StartAhead, FinalValue)
#Result.head()



























#TotalTrainX2.index=range(0,TotalTrainX2.shape[0])
#TotalTestX2.index=range(0,TotalTestX2.shape[0])
#TotalTrainY.index=range(0,TotalTrainY.shape[0])
#
##TotalTrainX2=TotalTrainX.iloc[:,Selection]
##TotalTestX2=TotalTestX.iloc[:,Selection]
#
#
#TotalTestX2=TotalTestX2[TotalTestX2["Ahead"]<6]
#Index=np.where(TotalTrainX2["Ahead"]<6)
#TotalTrainX2=TotalTrainX2[TotalTrainX2["Ahead"]<6]
#TotalTrainY=TotalTrainY.loc[Index]


#https://www.wunderground.com/history/airport/LEBL/2016/1/5/DailyHistory.html?req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&MR=1




#Selection=[ 48,   0, 178, 156, 166, 172,  44, 176, 159,  73, 136,  60,  11,
#           162,   3, 168,  86, 126,  30, 181, 116, 128, 146, 119,  43,  72,
#           6, 164,  78, 141,   1, 182, 138,  62,  36, 113, 158,  28,  46,
#           142, 102,  22,  76,  38,  58, 151, 131, 123,  52, 124, 110, 101,
#           51, 133,  53,  42, 103, 161,   9,  13, 173,  20, 148,  81, 150,
#           82,  68,  31, 183, 122,  16,  96,  91, 130, 144,   4, 153,  26, 12,  33]
#AheadLoc=np.where(TotalTrainX.columns=="Ahead")[0][0]
#Selection.append(AheadLoc)
#HourLoc=np.where(TotalTrainX.columns=="Hour")[0][0]
#Selection.append(HourLoc)
##DemandLoc=np.where(TotalTrainX.columns=="Demand")[0][0]
##Selection.append(DemandLoc)
##Selection=np.unique(np.array(Selection))



#BeforePrice = RawPrice.copy()
#del RawPrice["DayOnly"]
#BeforePrice.rename(columns={'Price':'AheadPrice',
#                            'prediction_date':'available_date2'}, inplace=True)
#
### Daily Average Price
#AveragePrice=pd.pivot_table(BeforePrice,values="AheadPrice",index=["DayOnly"],aggfunc=np.mean)
#AveragePrice=pd.DataFrame(AveragePrice).reset_index()
#AveragePrice.rename(columns={'AheadPrice': 'AveragePrice'}, inplace=True)
#AveragePrice["available_date2"]=pd.to_datetime(AveragePrice["DayOnly"]+" 23:00:00")
#
#DataUpgrade = pd.merge(WeatherData,BeforePrice[["available_date2","AheadPrice"]],how="left",on="available_date2")
#DataUpgrade2 = pd.merge(DataUpgrade,AveragePrice[["available_date2","AveragePrice"]],how="left",on="available_date2")




#Price = np.array(RawPrice.Price)
#Temp = Price.reshape(24,-1)
##DailyMean=Temp.mean(axis=1)
#DailyMean=np.zeros(24)
#Trend = np.tile(DailyMean,Temp.shape[1])
#Trend= pd.DataFrame(Trend,columns=['Mean'])
#RawPrice["Price"]=RawPrice["Price"] - Trend["Mean"]




#    ## Before Price
#    BeforePrice = RawPrice[["prediction_date","Price","Hour"]].copy()
#    BeforePrice["prediction_date"]=BeforePrice["prediction_date"] + datetime.timedelta(days=1)
#    BeforePrice.rename(columns={'Price':'AheadPrice'}, inplace=True)
#    BeforePrice=BeforePrice[BeforePrice.Hour==23]
#    BeforePrice["Year"]=pd.DatetimeIndex(BeforePrice["prediction_date"]).year
#    BeforePrice["Month"]=pd.DatetimeIndex(BeforePrice["prediction_date"]).month
#    BeforePrice["Day"]=pd.DatetimeIndex(BeforePrice["prediction_date"]).day
#
#
#    ## Daily Average Price
#    AveragePrice=pd.pivot_table(RawPrice,values="Price",index=["DayOnly"],aggfunc=np.mean)
#    AveragePrice=pd.DataFrame(AveragePrice).reset_index()
#    AveragePrice.rename(columns={'Price': 'AveragePrice'}, inplace=True)
#    AveragePrice["DayOnly"]=AveragePrice["DayOnly"] + datetime.timedelta(days=1)
#    AveragePrice["Year"]=pd.DatetimeIndex(AveragePrice["DayOnly"]).year
#    AveragePrice["Month"]=pd.DatetimeIndex(AveragePrice["DayOnly"]).month
#    AveragePrice["Day"]=pd.DatetimeIndex(AveragePrice["DayOnly"]).day
#
#
#    AdditionalPrice=pd.merge(BeforePrice[["Year","Month","Day","Hour","AheadPrice"]],
#                             AveragePrice[["Year","Month","Day","AveragePrice"]],on=["Year","Month","Day"])
#    Price = pd.merge(RawPrice,AdditionalPrice,on=["Year","Month","Day","Hour"])
#    del Price["DayOnly"]

