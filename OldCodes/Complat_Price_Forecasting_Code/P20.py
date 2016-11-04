import numpy as np
import pandas as pd
import os
import sys
from sys import path
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pdb, traceback
import datetime
#from svmutil import *

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


plt.rcParams['figure.figsize'] = (14.0, 8.0)
plt.ion()


# Options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 50000)
#pd.set_option('display.height', 50000)


#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def ParameterGen(Selection):
    if Selection=='Load':
        # Internal Parameter
        Param={'CVN': 5}
        Param['MachineList']=["Reg"]  # ,"GBM"
        Param['NGroup']=[1]
        Param['Ensem']="Simple"
        Param['GBM_NTry']=1000
        Param['GBM_LRate']=0.01
        Param['RF_NTree']=3000  # 200
        Param['C']=10
        Param['Gamma']=0.001
        Param['Epsilon']=0.01
        Param['Reg_Reg']=[0.1,1,2,3,5,10,15,20,30,40,50,100,200,300,500,1000]
        Param['Reg_CVN']=5
        Param['Reg_CVP']=0.4
        Param['Min']=0
        Param['Max']=700000

    elif Selection=='Price':
        Param={'CVN': 5}
        Param['MachineList']=["Reg"] # ,","Ridge"
        Param['NGroup']=[1]
        Param['Ensem']="Simple"
        Param['GBM_NTry']=1000
        Param['GBM_LRate']=0.02
        Param['RF_NTree']=3000  # 200
        Param['C']=2 # 78
        Param['Gamma']=0.001
        Param['Epsilon']=0.05  # 3.1
        Param['Reg_Reg']=[0.1,1,2,3,5,10,15,20,30,40,50,100,200,300,500,1000]
        Param['Reg_CVN']=5
        Param['Reg_CVP']=0.4
        Param['Min']=0
        Param['Max']=700000

    return Param

#######################################################################
## Price Reader
#######################################################################
def PriceReader(PriceFileLoc,PriceFileName):
    RawPrice  = pd.read_csv(PriceFileLoc + PriceFileName)
    RawPrice["prediction_date"]=pd.to_datetime(RawPrice["prediction_date"])
    return RawPrice


#######################################################################
## Weather Data Reading
#######################################################################
def SingleWeather(WeatherFileLoc,WeatherRange,WindSelection):

    try:
        os.remove(WeatherFileLoc+'.ds_store')
    except OSError:
        pass

    FileList=os.listdir(WeatherFileLoc)
    FileList.sort()

    for f in range(0,len(FileList)):
        ReadData  = pd.read_csv(WeatherFileLoc + FileList[f])

        ReadData["WindSquareA"]=ReadData["wind_speed_100m"]**2
        ReadData["WindSquareB"]=ReadData["wind_speed_100m"]**3

        ReadData["WindSquareC"]=ReadData["wind_speed"]**2
        ReadData["WindSquareD"]=ReadData["wind_speed"]**3

        ReadData["WindSquareE"]=ReadData["wind_gust"]**2
        ReadData["WindSquareF"]=ReadData["wind_gust"]**3


        Single = ReadData[ ReadData.point==2]
        Single = Single[np.append(TimeIndex,WindSelection)]

        Twin = ReadData[ ReadData.point==3]
        Twin = Twin[np.append(TimeIndex,WindSelection)]

        Single = pd.merge(Single,Twin,on=TimeIndex,suffixes=('2','3'))

        for t in WeatherRange:
            Temp2 = ReadData[ ReadData.point==t]
            Temp2 = Temp2[np.append(TimeIndex,WindSelection)]   #  <----
            Single = pd.merge(Single,Temp2,on=TimeIndex,suffixes=('',str(t)))

        ## Ahead Generation
        Single["Ahead"] = np.repeat(range(1,8),24)

        if f==0:
            Data = Single
        else:
            Data = pd.concat([Data,Single])

    Data.index=range(0,Data.shape[0])
    Data["prediction_date"]=pd.to_datetime(Data["prediction_date"])
    del Data["available_date"]
#    Data["available_date"]= pd.DatetimeIndex(Data["available_date"]).date
#    Data["available_date"]=pd.to_datetime(Data["available_date"])
#
#    Data["WeekDay"]=pd.DatetimeIndex(Data["prediction_date"]).weekday
#    Data["Year"]=pd.DatetimeIndex(Data["prediction_date"]).year
#    Data["Month"]=pd.DatetimeIndex(Data["prediction_date"]).month
#    Data["Day"]=pd.DatetimeIndex(Data["prediction_date"]).day
#    Data["Hour"]=pd.DatetimeIndex(Data["prediction_date"]).hour
#
#    Data["Prediction_DayOnly"]= pd.DatetimeIndex(Data["prediction_date"]).date
#    Data["Prediction_DayOnly"]= pd.to_datetime(Data["Prediction_DayOnly"])

    return Data



#######################################################################
## Wind Reader
#######################################################################
def WindReader(FileName):
    Wind = pd.read_csv(WindPowerLoc + FileName)  # PTHourlyDemand.csv  PDEnergy.csv
    Wind = Wind[["Date","Year","Month","Day","Hour","Minute","Wind"]]
    Wind["Date"]=pd.to_datetime(Wind["Date"])
    TimeIndex=["Date","Year","Month","Day","Hour"]
    AverageWind = pd.pivot_table(Wind,values="Wind",index=TimeIndex,aggfunc=np.mean)
    AverageWind = pd.DataFrame(AverageWind).reset_index()
    AverageWind.rename(columns={'Date': 'prediction_date'},inplace=True)
    return AverageWind




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
    #PTGen['Gen'] = PTGen.iloc[:,range(6,20)].sum(axis=1) forecast_demand

    TimeIndexPT=["Date","Year","Month","Day","Hour"]
    PTAverageGen = pd.pivot_table(PTGen,values="real_demand",index=TimeIndexPT,aggfunc=np.mean)
    PTAverageGen=pd.DataFrame(PTAverageGen).reset_index()
    PTAverageGen.rename(columns={'real_demand': 'Demand'}, inplace=True)
    PTAverageGen.rename(columns={'Date': 'prediction_date'},inplace=True)
    return PTAverageGen

# forecast_demand real_demand


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
    TotalTrainX1.rename(columns={'Ahead': 'Ahead2','Hour': 'Hour2'},inplace=True)

    TotalTrainX3=pd.concat([TotalTrainX2.iloc[1:,:],TotalTrainX2.iloc[TempNumber0:TempNumber1,:]],axis=0)
    TotalTrainX3.index=range(0,TotalTrainX3.shape[0])
    TotalTrainX3.rename(columns={'Ahead': 'Ahead2','Hour': 'Hour2'},inplace=True)

    TotalTrainX=pd.concat([TotalTrainX1, TotalTrainX2,TotalTrainX3],axis=1)
    LocAhead=np.where(TotalTrainX.columns=="Ahead")[0][0]
    LocHour=np.where(TotalTrainX.columns=="Hour")[0][0]
    TotalTrainX.columns=range(0,TotalTrainX.shape[1])
    TotalTrainX.rename(columns={TotalTrainX.columns[LocAhead]:"Ahead",
                                TotalTrainX.columns[LocHour]:"Hour"},inplace=True)


    TotalTestX1=pd.concat([TotalTestX2.iloc[0:1,:], TotalTestX2.iloc[:-1,:]],axis=0)
    TotalTestX1.index=range(0,TotalTestX1.shape[0])
    TotalTestX1.rename(columns={'Ahead': 'Ahead2','Hour': 'Hour2'},inplace=True)
    TotalTestX3=pd.concat([TotalTestX2.iloc[1:,:], TotalTestX2.iloc[TempNumber0:TempNumber1,:]],axis=0)
    TotalTestX3.index=range(0,TotalTestX3.shape[0])
    TotalTestX3.rename(columns={'Ahead': 'Ahead2','Hour': 'Hour2'},inplace=True)

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
    TargetAhead=range(1,8)

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
    Normal=Unique.sub(Unique.mean(1), axis=0).div(Unique.std(1), axis=0)
    #Normal=(Unique - Unique.mean()) / Unique.std()

    # Set data
    TrainX=Normal.iloc[0:TrainX.shape[0],:]
    TestsX=Normal.iloc[TrainX.shape[0]:,:]

    return TrainX, TestsX



#########################################################################
## Cross Validation Function [ General Case ]
#######################################################################
def CrossVal(TrainX,TrainY,Param):

    CVN=Param['CVN']
    MachineList=Param['MachineList']

    kf = KFold( len(TrainX) ,shuffle=True, n_folds=CVN )
    FunRMSE=np.zeros((len(MachineList)))
    MachineValue=np.zeros((len(MachineList),len(TrainX)))


    # Preprocessing
    STD=np.std(TrainX,axis=0)
    NonZero=np.nonzero(STD)[0]
    Unique=TrainX.iloc[:,NonZero]
    #TrainX=(Unique - Unique.mean()) / Unique.std()
    TrainX=Unique.sub(Unique.mean(1), axis=0).div(Unique.std(1), axis=0)

    # Outlier Detection
    #SubTrainX,SubTrainY=OutlierDetector(SubTrainX,SubTrainY)

    for machindex in range(0,len(MachineList)):
        Machine=MachineList[machindex]

        for train_index, test_index in kf:
            SubTrainX = TrainX.iloc[train_index,:]
            SubTestsX = TrainX.iloc[test_index,:]

            SubTrainY = TrainY.iloc[train_index]
            SubTestsY = TrainY.iloc[test_index]

            SubTrainX.index=np.arange(0,len(SubTrainX))
            SubTestsX.index=np.arange(0,len(SubTestsX))

            SubTrainY.index=np.arange(0,len(SubTrainY))
            SubTestsY.index=np.arange(0,len(SubTestsY))

            # Forecast
            Forecast=Machinery(Machine,SubTrainX,SubTrainY,SubTestsX,Param)
            MachineValue[machindex,test_index]=Forecast

        TempPerformance=PerMeasure(TrainY,MachineValue[machindex,])
        FunRMSE[machindex]=TempPerformance
        #print Machine, TempPerformance

    print '    '
    print FunRMSE
    print '    '

    return FunRMSE


#######################################################################
## Performance Measure [ Geleral Case ]
#######################################################################
def PerMeasure(X,Y):
    P=np.mean(np.absolute( X - Y ))
    return P


#######################################################################
## Post Processing  [ General Case ]
#######################################################################
def PostProcessing(Prediction,Param):
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
def OptimalReg(TrainX,TrainY,Param):

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
            NormOutput=PostProcessing(Output,Param)
            SubPerformance[CVindex]=PerMeasure(NormOutput,SubTestsY.values)

        Performance[Alphaindex]=SubPerformance.mean()

    BestReg=AlphaList[np.argmin(Performance)]
    #print "                Best Regulator: ", BestReg, " Performance: ", np.min(Performance)
    return BestReg






########################################################################
## Ridge Regerssion
########################################################################
def RidgeReg(TrainX,TrainY,Param):
    BestReg=OptimalReg(TrainX,TrainY,Param)	# Optimal Alpha Detection
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






#################################################################################
## KNN
#################################################################################
def KNearestNeighbors(TrainX , TrainY):
    NNeighbors=Param['NNeighbors']
    KNN=KNeighborsRegressor(n_neighbors=NNeighbors,p=2,algorithm='auto',
                            leaf_size=300,weights='uniform') # ,weights='distance'

    return KNN




#######################################################################
## Gradient Boosting Training
#######################################################################
def GradientBoosting(TrainX,TrainY,Param):

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
    GBM = GradientBoostingRegressor(loss='lad',n_estimators=Param['GBM_NTry'],
                                    max_features=TrainX.shape[1]/6,
                                    max_depth=5,
                                    learning_rate=Param['GBM_LRate'],subsample=0.5,verbose=0)

    return GBM






#################################################################################
#	## Random Forest
#################################################################################
def RandomForest(TrainX,TrainY,Param):

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
                             bootstrap=True,max_features=TrainX.shape[1]/4,
                             n_jobs=-1)

    # min_samples_leaf=1,
    #
    # min_samples_split
    #

    return RF




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
## Support Vector Machine Regression
#################################################################################
def SupportVectorMachine(TrainX,TrainY,Param):

    # CV Data Spliting
#    SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])

#    # Cross validation to select the best C
#    CandidateC=Param['C']
#    Score=np.zeros(len(CandidateC))
#    for i in range(0,len(CandidateC)):
#        Support=SVR(C=CandidateC[i], kernel='rbf', degree=3,gamma=Param['Gamma'],shrinking=True,tol=CandidateC[i]/1000.0,cache_size=1000,verbose=0)
#        Support.fit(SubTrainX,SubTrainY)
#        TestOutput=Support.predict(SubTestsX)
#        NewPrediction=PostProcessing(TestOutput)
#        Score[i]=PerMeasure(NewPrediction,SubTestsY)
#
#    BestCIndex=np.argmin(Score)
#    BestC=CandidateC[BestCIndex]
#    BestPerformance=np.min(Score)
#
#    print "BestC: ", BestC, " Best Performance: ", BestPerformance

    # Final Prediction


    Support=SVR(C=Param['C'], kernel='rbf', degree=3, epsilon=Param['Epsilon'],
                gamma=Param['Gamma'], shrinking=True,verbose=0)
    #tol=CandidateC[i]/1000.0
    # cache_size=200,
    return Support



#################################################################################
## SVM Lib
#################################################################################
def SVMLIB(TrainX,TrainY,TestsX,Param):
    Mean=TrainY.mean()
    TrainY = TrainY - Mean
    STD = TrainY.std()
    TrainY = TrainY / STD
    prob  = svm_problem(TrainY.values.tolist(), TrainX.values.tolist())
    SVMParam='-s 3 -t 2 -c ' + str(Param['C']) + ' -g ' + str(Param['Gamma']) + ' -p ' + str(Param['Epsilon']) + ' -b 0 -q'
    param = svm_parameter(SVMParam)
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(TestsX.iloc[:,1].values.tolist(), TestsX.values.tolist(), m, '-b 0 -q')
    Output=np.array(p_label)  * STD + Mean
    return Output


################################################################################
## Main Machinenary Selection
################################################################################
def Machinery(MachineInfo,TrainX,TrainY,TestsX,Param):
    Mean=TrainY.mean()
    TrainY = TrainY - Mean
    STD = TrainY.std()
    TrainY = TrainY / STD


    if MachineInfo=="Reg":
        Attribute=LinearReg(TrainX,TrainY)

    elif MachineInfo=='Ridge':
        Attribute=RidgeReg(TrainX,TrainY,Param)

    elif MachineInfo=='RF':
        Attribute=RandomForest(TrainX,TrainY,Param)

    elif MachineInfo=='GBM':
        Attribute=GradientBoosting(TrainX,TrainY,Param)

    elif MachineInfo=='NN':
        Temp=1

    elif MachineInfo=='SVM':
        Attribute=SVR(C=Param['C'], kernel='rbf', degree=3, epsilon=Param['Epsilon'],
                    gamma=Param['Gamma'], shrinking=True,verbose=0)
        #Attribute=SupportVectorMachine(TrainX,TrainY)

        Attribute.fit(TrainX,TrainY)
        TestOutput=Attribute.predict(TestsX)
        Output=np.array(TestOutput)  * STD + Mean
        return Output

    elif MachineInfo=='SVMLIB':
         TestOutput = SVMLIB(TrainX,TrainY,TestsX,Param)
         return TestOutput

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
    TestOutput=np.array(TestOutput)  * STD + Mean
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
    NormForecast=PostProcessing(TempResult,Param)
    PrintOutRMSE=PerMeasure(TrainY,NormForecast)
    print FunRMSE, PrintOutRMSE

    return NormForecast, Weight, BestGroup, BestMachine



#################################################################################
## Quick Ensemble
##################################################################################
def QuickEnsemble(Param,FunRMSE):
    if Param['Ensem']=='Best':
        GameSize=len(Param['NGroup']) * len(Param['MachineList'])
        Weight=np.ones(1)*1.0
        BestMachine=np.where(FunRMSE==np.min(FunRMSE))[0][0]
        BestMachine=np.array(BestMachine)

    elif Param['Ensem']=='Simple':
        GameSize=len(Param['NGroup']) * len(Param['MachineList'])
        Weight=np.ones(GameSize)*1.0/GameSize
        Weight=Weight[np.newaxis].transpose()
        BestMachine=range(0,len(Param['MachineList']))

    elif Param['Ensem']=='Weight':
        InverseWeight=1.0 / FunRMSE
        Weight=InverseWeight / sum(InverseWeight)
        Weight=Weight[np.newaxis].transpose()
        BestMachine=range(0,len(Param['MachineList']))

    return Weight,BestMachine



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
def FinalTraining(TrainX,TrainY,TestsX,Weight,BestMachine,Param):

    MachineList=Param['MachineList']
    SavePredict=np.zeros( ( len(BestMachine) , len(TestsX) ) )
    for machindex in range(0,len(BestMachine)):
        Machine=MachineList[BestMachine[machindex]]

        # Outlier Detection
        #TrnX,TrnY=OutlierDetector(TrnX,TrnY)

        # Preprocessing
        TrainX,TestsX=Standardization(TrainX, TestsX)

        # Forecast
        Forecast=Machinery(Machine,TrainX,TrainY,TestsX,Param)
        SavePredict[machindex,:]=Forecast

    # Weighted Output Generation
    TempResult= Weight * SavePredict
    WeightedOutput=np.sum(TempResult,axis=0)
    WeightedOutput=PostProcessing(WeightedOutput,Param)

    return WeightedOutput





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##    Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def WeatherPriceCombine(TotalWeather,Price):

    WeatherPrice=pd.merge(TotalWeather, Price,how="left",on=["prediction_date","Year","Month","Day","Hour"])

    BeforePrice = Price.copy()
    BeforePrice["DayOnly"]= pd.DatetimeIndex(BeforePrice["prediction_date"]).date
    BeforePrice["DayOnly"]=pd.to_datetime(BeforePrice["DayOnly"])

    BeforePriceA=BeforePrice[BeforePrice.Hour==22]
    BeforePriceA.rename(columns={'Price':'AheadPriceA'}, inplace=True)

#    BeforePriceA=BeforePrice[BeforePrice.Hour==22]
#    BeforePriceA.rename(columns={'Price':'AheadPriceA'}, inplace=True)

    BeforePriceB=BeforePrice[BeforePrice.Hour==21]
    BeforePriceB.rename(columns={'Price':'AheadPriceB'}, inplace=True)

#    BeforePriceC=BeforePrice[BeforePrice.Hour==20]
#    BeforePriceC.rename(columns={'Price':'AheadPriceC'}, inplace=True)
#
#    BeforePriceD=BeforePrice[BeforePrice.Hour==19]
#    BeforePriceD.rename(columns={'Price':'AheadPriceD'}, inplace=True)
#
#    BeforePriceE=BeforePrice[BeforePrice.Hour==18]
#    BeforePriceE.rename(columns={'Price':'AheadPriceE'}, inplace=True)
#
#    BeforePriceF=BeforePrice[BeforePrice.Hour==17]
#    BeforePriceF.rename(columns={'Price':'AheadPriceF'}, inplace=True)
#
#    BeforePriceG=BeforePrice[BeforePrice.Hour==16]
#    BeforePriceG.rename(columns={'Price':'AheadPriceG'}, inplace=True)


    ## Daily Average Price
    AveragePrice=pd.pivot_table(BeforePrice,values="Price",index=["DayOnly"],aggfunc=np.mean)
    AveragePrice=pd.DataFrame(AveragePrice).reset_index()
    AveragePrice.rename(columns={'Price': 'AveragePrice'}, inplace=True)

    DataUpgrade = pd.merge(WeatherPrice,BeforePriceA[["DayOnly","AheadPriceA"]],how="left",on="DayOnly")
    DataUpgrade = pd.merge(DataUpgrade,BeforePriceB[["DayOnly","AheadPriceB"]],how="left",on="DayOnly")
#    DataUpgrade = pd.merge(DataUpgrade,BeforePriceC[["DayOnly","AheadPriceC"]],how="left",on="DayOnly")
#    DataUpgrade = pd.merge(DataUpgrade,BeforePriceD[["DayOnly","AheadPriceD"]],how="left",on="DayOnly")
#    DataUpgrade = pd.merge(DataUpgrade,BeforePriceE[["DayOnly","AheadPriceE"]],how="left",on="DayOnly")
#    DataUpgrade = pd.merge(DataUpgrade,BeforePriceF[["DayOnly","AheadPriceF"]],how="left",on="DayOnly")
#    DataUpgrade = pd.merge(DataUpgrade,BeforePriceG[["DayOnly","AheadPriceG"]],how="left",on="DayOnly")
#
    DataUpgrade2 = pd.merge(DataUpgrade,AveragePrice[["DayOnly","AveragePrice"]],how="left",on="DayOnly")


    DataUpgrade2 = DataUpgrade2.sort(TimeIndex,ascending=[1,1])

    return WeatherPrice



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##         Load Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def LoadForecasting(WeatherDemand,Threshold,Indicator):

    # TestX Generation
    DemandTestX = WeatherDemand[WeatherDemand.prediction_date > Threshold]
    NextGenTest = DemandTestX[TimeIndex]
    DemandTestX = DemandTestX.drop(["Demand","available_date","DayOnly","prediction_date"],1)
    DemandTestX.index=range(0,DemandTestX.shape[0])


    DemandTrainX=WeatherDemand[WeatherDemand.prediction_date <= Threshold]
    NextGenTrain = DemandTrainX[["Demand","available_date","prediction_date"]]

    DemandTrainX=DemandTrainX[~(DemandTrainX.isnull().any(axis=1))]
    DemandTrainX.index=range(0,DemandTrainX.shape[0])


    AvailableDate=DemandTrainX[["prediction_date","available_date","DayOnly","Ahead"]]
    AvailableDate["PredictionDayOnly"]=pd.DatetimeIndex(AvailableDate["prediction_date"]).date
    AvailableDate["PredictionDayOnly"]=pd.to_datetime(AvailableDate["PredictionDayOnly"])

    # TrainY Generation
    DemandTrainY=DemandTrainX["Demand"]
    DemandTrainY.index=range(0,DemandTrainY.shape[0])
    DemandTrainX = DemandTrainX.drop(["Demand","available_date","DayOnly","prediction_date"],1)

    # Training Data Expansion
    if Indicator==1:
        DemandTrainX, DemandTestX = TrainingExtention(DemandTrainX, DemandTestX)

    # Internal Parameter
    Param2=ParameterGen('Load')

    ## Main Game Starts
    FunRMSE=0
    #FunRMSE=CrossVal(AvailableDate,DemandTrainX,DemandTrainY,Param2)

    Weight, BestGroup, BestMachine = QuickEnsemble(Param2,FunRMSE)

    LoadForecast=FinalTraining(DemandTrainX,DemandTrainY,DemandTestX,Weight,BestGroup,BestMachine,Param2)

    NextGenTest["Demand"]=LoadForecast


    # Final Value Generation
    UniqueNextGen=NextGenTest["prediction_date"].drop_duplicates()

    for i in range(0,len(UniqueNextGen)):
        TargetTime=UniqueNextGen.iloc[i]
        Sub=NextGenTest[NextGenTest.prediction_date==TargetTime]
        Sub=Sub[["prediction_date","Demand"]]
        AverageForecast=pd.pivot_table(Sub,values="Demand",index=["prediction_date"],aggfunc=np.mean)
        AverageForecast=pd.DataFrame(AverageForecast).reset_index()

        if i==0:
            SavingAverage=AverageForecast
        else:
            SavingAverage=pd.concat([SavingAverage,AverageForecast])

    NextGenTest=pd.merge(NextGenTest[TimeIndex],SavingAverage,on=["prediction_date"])
    DemandSupport=pd.concat([NextGenTrain,NextGenTest],axis=0)

    DemandSupport = DemandSupport.sort(TimeIndex,ascending=[1,1])
    DemandSupport.index =range(0,DemandSupport.shape[0])

    plt.figure()
    plt.plot(SavingAverage.Demand,'rs-')
    plt.show()





    #NewTrainX, NewTestsX = SecondWave(TrainOutput,LoadForecast,DemandTrainX,DemandTestX)
    #
    #FunRMSE2, LoadPredict2 = CrossVal(NewTrainX,DemandTrainX)
    #
    #TrainOutput2,Weight2,BestGroup2,BestMachine2=Ensemble(FunRMSE2,LoadPredict2,DemandTrainY)
    #
    #LoadForecast2=FinalTraining(NewTrainX,DemandTrainX,NewTestsX,Weight2, BestGroup2, BestMachine2)

    return DemandSupport




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##         Training and Testing Data Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def TrainAndTestGen(WeatherPrice,PriceThreshold):
    # TestX
    TotalTestX=WeatherPrice[WeatherPrice.prediction_date > PriceThreshold]
    Tracker=TotalTestX[TimeIndex]
    TotalTestX.index=range(0,TotalTestX.shape[0])

    # TrainX
    TotalTrainX=WeatherPrice[(WeatherPrice.prediction_date <= PriceThreshold) ]
    TotalTrainX=TotalTrainX[~(TotalTrainX.isnull().any(axis=1))]
    TotalTrainX.index=range(0,TotalTrainX.shape[0])

    # TrainY
    TotalTrainY=TotalTrainX["Price"]
    TotalTrainY.index=range(0,TotalTrainY.shape[0])

    # Available
    AvailableDate=TotalTrainX[["prediction_date","available_date","DayOnly","Ahead"]]
    AvailableDate["PredictionDayOnly"]=pd.DatetimeIndex(AvailableDate["prediction_date"]).date
    AvailableDate["PredictionDayOnly"]=pd.to_datetime(AvailableDate["PredictionDayOnly"])

    # Remove String Values
    TotalTrainX = TotalTrainX.drop(["Price","available_date","DayOnly","prediction_date"],1)
    TotalTestX = TotalTestX.drop(["Price","available_date","DayOnly","prediction_date"],1)

    return TotalTrainX, TotalTrainY, TotalTestX, AvailableDate, Tracker



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##         Final Result Averaging
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def FinalTouch(Tracker,SaveForecast):
    Tracker["Forecast"]=SaveForecast
    UniqueNextGen=Tracker["prediction_date"].drop_duplicates()

    for i in range(0,len(UniqueNextGen)):
        TargetTime=UniqueNextGen.iloc[i]
        Sub=Tracker[Tracker.prediction_date==TargetTime]
        Sub=Sub[Sub.available_date==Sub.available_date.max()]
        Sub=Sub[["prediction_date","Forecast"]]
        AverageForecast=pd.pivot_table(Sub,values="Forecast",index=["prediction_date"],aggfunc=np.mean)
        AverageForecast=pd.DataFrame(AverageForecast).reset_index()

        if i==0:
            SavingAverage=AverageForecast
        else:
            SavingAverage=pd.concat([SavingAverage,AverageForecast])

    return  SavingAverage




########################################################################
# Printing
########################################################################
def PostPrinting(TargetDateTime, SubmissionDate, Prediction):
    # Prepare to print it out
    SubmissionFolder="/Users/dueheelee/Documents/ComPlatt/Submission/"
    ForecastorID = "4c17235d2f125cb17cb2d782fce8d201"
    StartPoint = 0 #(StartAhead-1)*24
    SelectForecast = Prediction[ StartPoint : (StartPoint + 120)]

    Seed=np.array(range(0,24))
    Seed=np.tile(Seed,5)
    Seed=np.append([22,23],Seed)
    Seed=Seed[:-2]


    Template=pd.DataFrame(Seed,columns=["hour"])
    Template["forecaster"]=ForecastorID

    Available = datetime.datetime.strptime(SubmissionDate, "%Y-%m-%d")
    AvailableString = Available.strftime('%d-%m-%Y')
    Template["availabledate"] = AvailableString
    Template["predictiondate"] = AvailableString

    PredictionDate=['a']*5
    for i in range(0,5):
        TempDate = Available + datetime.timedelta(days=(i+1))
        PredictionDate[i]=TempDate.strftime('%d-%m-%Y')

    Temp = np.repeat(PredictionDate,24)
    Template.loc[2:,"predictiondate"]= Temp[:-2]
    Template["value"] = SelectForecast

    Template=Template[["forecaster","availabledate","predictiondate","hour","value"]]

    Template.to_csv(SubmissionFolder+ForecastorID+"_"+SubmissionDate+".csv",
                    index=False,heater=True,date_format="%d-%m-%Y")
    return Template





    # wind_speed_100m	wind_direction_100m
    # temperature	air_density	pressure	precipitation	wind_gust
    # radiation 	wind_speed 	wind_direction

































#ESWind = WindReader("ESWind.csv")
#ESWindThreshold=Price.loc[ESWind.shape[0]-1,"prediction_date"]

#PTWind = WindReader("PTWind.csv")
#PTWindThreshold=Price.loc[PTWind.shape[0]-1,"prediction_date"]



#OldWeather = OldWeatherRead(OldWeatherLoc, "historical_weather.csv",WindSelection,WindSelection)















# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Parameters
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
SubmissionDate = "2016-04-14"
TargetDate=pd.to_datetime(SubmissionDate).date()
WindPowerLoc    = "/Users/dueheelee/Documents/ComPlatt/Data/WindPower/"
PriceFileLoc    = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Price/"
DemandLoc       = "/Users/dueheelee/Documents/ComPlatt/Data/Demand/"
OldWeatherLoc   = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Weather/"
WeatherFileLoc  = "/Users/dueheelee/Documents/ComPlatt/Data/Forecast_Weather/"
ID = ["prediction_date","Year","Month","Day","Hour"]
TimeIndex = ["available_date","prediction_date"]

# Answer
Answer=np.array([24.39,21.5,20.69,20.45,20.34,21.08,25,25.22,25.9,29.69,30.69,29.96,28,25.73,23.19,
                 21.74,21.32,21.49,22.73,23.37,26,26.42,25,24.69])

WeatherRange=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
WindSelection =["WindSquareA","WindSquareB","WindSquareC","WindSquareD",
                "temperature","wind_speed_100m","wind_speed"]


#"WindSquareA","WindSquareB","WindSquareC","WindSquareD","air_density","pressure",,"wind_gust","precipitation"
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Price
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Price = PriceReader(PriceFileLoc,"NewPrice.csv")
PriceThreshold=Price.loc[Price.shape[0]-1,"prediction_date"]

## Demand
ESAverageGen = ESGenerationReader(DemandLoc,"ESDemand.csv")
DemandThreshold=ESAverageGen.loc[ESAverageGen.shape[0]-1,"prediction_date"]

## Energy
PTAverageGen = pd.read_csv(DemandLoc + "PTHourlyDemand.csv")  # PTHourlyDemand.csv  PDEnergy.csv
PTAverageGen["prediction_date"]=pd.to_datetime(PTAverageGen["prediction_date"])

## Weather
Weather = SingleWeather(WeatherFileLoc,WeatherRange,WindSelection)





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#####         Load Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
## Data Gen

TotalWeather=[0]*5
for G in range(1,6):   # 6
    Temp=Weather[Weather.Ahead==G]
    del Temp["Ahead"]
    for B in range(G+1,8):
        SubWeather=Weather[Weather.Ahead==B]
        del SubWeather["Ahead"]
        Temp = pd.merge(Temp,SubWeather,on="prediction_date",suffixes=('',B))

    Temp=pd.merge(Temp ,PTAverageGen[["prediction_date","DemandPT"]],how="left",on=["prediction_date"])
    TotalWeather[G-1]=Temp


for G in range(0,5):
    WeatherDemand=pd.merge(TotalWeather[G],ESAverageGen[["prediction_date","Demand"]],how="left",
                           on=["prediction_date"])
    WeatherDemand["Hour"]=pd.DatetimeIndex(WeatherDemand["prediction_date"]).hour

    # TestX Generation
    #DemandTestX = WeatherDemand[WeatherDemand.prediction_date > DemandThreshold]
    DemandTestX =WeatherDemand[pd.isnull(WeatherDemand["Demand"])]
    DemandTestX.index=range(0,DemandTestX.shape[0])
    NextGenTest=DemandTestX[["prediction_date","Demand"]]
    DemandTestX = DemandTestX.drop(["Demand","prediction_date"],1)


    DemandTrainX=WeatherDemand[~pd.isnull(WeatherDemand["Demand"])]
    #DemandTrainX=WeatherDemand[WeatherDemand.prediction_date <= DemandThreshold]
    DemandTrainX.index=range(0,DemandTrainX.shape[0])

    DemandTrainY = DemandTrainX["Demand"]
    DemandTrainX = DemandTrainX.drop(["Demand","prediction_date"],1)

    # Internal Parameter
    Param2=ParameterGen('Load')
    FunRMSE = CrossVal(DemandTrainX,DemandTrainY,Param2)
    Weight, BestMachine = QuickEnsemble(Param2,FunRMSE)
    LoadForecast=FinalTraining(DemandTrainX,DemandTrainY,DemandTestX,Weight,BestMachine,Param2)
    NextGenTest["Demand"]=LoadForecast

    if G==0:
        LoadSaver=NextGenTest
    else:
        LoadSaver = pd.concat([LoadSaver,NextGenTest])
        LoadSaver=LoadSaver.drop_duplicates("prediction_date")

LoadSaver = LoadSaver.sort("prediction_date",ascending=[1])
plt.plot(LoadSaver.Demand)

bb




#  Twin = Twin[np.append(TimeIndex,WindSelection)]
#
#        Single = pd.merge(Single,Twin,on=TimeIndex,suffixes=('2','3'))
#
#        for t in WeatherRange:
#            Temp2 = ReadData[ ReadData.point==t]
#            Temp2 = Temp2[np.append(TimeIndex,WindSelection)]   #  <----
#            Single = pd.merge(Single,Temp2,on=TimeIndex,suffixes=('',str(t)))
#
#        ## Ahead Generation
#        Single["Ahead"] = np.repeat(range(1,8),24)




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Wind Power Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Fast=1
if Fast==1:
    DemandSupport = pd.read_csv('Load.csv')
    DemandSupport["prediction_date"]=pd.to_datetime(DemandSupport["prediction_date"])
    DemandSupport["available_date"]=pd.to_datetime(DemandSupport["available_date"])

    ESWindSupport = pd.read_csv('W1.csv')
    ESWindSupport["prediction_date"]=pd.to_datetime(ESWindSupport["prediction_date"])
    ESWindSupport["available_date"]=pd.to_datetime(ESWindSupport["available_date"])

    PTWindSupport = pd.read_csv('W2.csv')
    PTWindSupport["prediction_date"]=pd.to_datetime(PTWindSupport["prediction_date"])
    PTWindSupport["available_date"]=pd.to_datetime(PTWindSupport["available_date"])

else:

#    WeatherRange=[1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #1 5 9 13 17
#    WindSelection =["temperature","pressure","wind_speed_100m","wind_speed","wind_gust","precipitation","radiation"]
#    WindSelection=WindSelection
#    OldWeather=OldWeatherRead(OldWeatherLoc, "historical_weather.csv",WindSelection,WindSelection)
#    NewWeather = SingleWeather(WeatherFileLoc,WindSelection,WindSelection)
#    TotalWeather=pd.concat([OldWeather,NewWeather])

    # Load Forecasting
    SubWeather=pd.merge(TotalWeather,Energy,how="left",on=ID)
    SubWeather = pd.merge(SubWeather,PTAverageGen,how="left",on=ID)
    DemandSupport = LoadForecasting(SubWeather,DemandThreshold,0)
    DemandSupport.to_csv('Load.csv',index_label=None,index=False)

    # ES Wind Forecasting
#    SubWeather=pd.merge(TotalWeather,ESWind,how="left",on=ID)
#    SubWeather.rename(columns={'Wind': 'Demand'}, inplace=True)
#    ESWindSupport = LoadForecasting(SubWeather,ESWindThreshold,0)
#    ESWindSupport.rename(columns={'Demand': 'ESWind'}, inplace=True)
#    ESWindSupport.to_csv('W1.csv',index_label=None,index=False)
#
#    # PT Wind Forecasting
#    SubWeather=pd.merge(TotalWeather,PTWind,how="left",on=ID)
#    SubWeather.rename(columns={'Wind': 'Demand'}, inplace=True)
#    PTWindSupport = LoadForecasting(SubWeather,PTWindThreshold,0)
#    PTWindSupport.rename(columns={'Demand': 'PTWind'}, inplace=True)
#    PTWindSupport.to_csv('W2.csv',index_label=None,index=False)



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherRange=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
WindSelection =["WindSquareA","WindSquareB","WindSquareC","WindSquareD",
                "wind_speed_100m","wind_speed"]
WindSelection=WindSelection #["temperature","radiation"]
OldWeather=OldWeatherRead(OldWeatherLoc, "historical_weather.csv",WindSelection,WindSelection)
NewWeather = SingleWeather(WeatherFileLoc,WindSelection,WindSelection)
TotalWeather=pd.concat([OldWeather,NewWeather])
MaxAhead = max(TotalWeather.Ahead)



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherPrice=WeatherPriceCombine(TotalWeather,Price)   # NewWeather TotalWeather
WeatherPrice=pd.merge(WeatherPrice ,DemandSupport,how="left",on=TimeIndex)
WeatherPrice=pd.merge(WeatherPrice ,Energy[["prediction_date","DemandPT"]],how="left",on=["prediction_date"])
#WeatherPrice=pd.merge(WeatherPrice ,ESWindSupport,how="left",on=TimeIndex)
#WeatherPrice=pd.merge(WeatherPrice ,PTWindSupport,how="left",on=TimeIndex)

WeatherPrice = WeatherPrice.sort(TimeIndex,ascending=[1,1])
#del WeatherPrice["Day"]
#del WeatherPrice["Year"]
#del WeatherPrice["WeekDay"]    ## Always Delete
#del WeatherPrice["Month"]    ## Always Delete


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Training and Testing Data Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
TotalTrainX, TotalTrainY, TotalTestX, AvailableDate, Tracker = TrainAndTestGen(WeatherPrice,PriceThreshold)




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Cross Validation Ticket Generation
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Param=ParameterGen('Price')
#Param['C']=1
#Param['Gamma']=0.001
#Param['Epsilon']=0.001
#Param['MachineList']=["RF","GBM","Ridge","Reg"]
#for k in range(0,4):
#    Param['C']=0.6
#    Weight, BestGroup, BestMachine = QuickEnsemble(Param,0)
#    SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine,Param)
#    SavingAverage1 = FinalTouch(Tracker,SaveForecast)
#    FinalValue=SavingAverage1.Forecast.values
#    MAE=np.absolute(FinalValue[:24]-Answer).mean()
#    print Param['C'], '  ', MAE


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Final Training 1
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Param=ParameterGen('Price')
FunRMSE = CrossVal(AvailableDate,TotalTrainX,TotalTrainY,Param)
Weight, BestGroup, BestMachine = QuickEnsemble(Param,FunRMSE)
SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine,Param)
SavingAverage1 = FinalTouch(Tracker,SaveForecast)
FinalValue=SavingAverage1.Forecast.values
MAE=np.absolute(FinalValue[:24]-Answer).mean()
print FunRMSE.mean() , '  /  ', MAE, '  /  ' , np.append(FunRMSE,MAE).mean()




t=range(0,24)
plt.plot(t,FinalValue[:24],'rs-',t,Answer,'b^-')
plt.show()











#
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###############          LoadForecasting Revisit
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#WeatherPrice2.loc[Tracker.index,"Price"]=SaveForecast
#del WeatherPrice2["Demand"]
#WeatherDemand2=pd.merge(WeatherPrice2,PTAverageGen,how="left",
# on=["prediction_date","Year","Month","Day","Hour"])
#
#DemandSupport2 = LoadForecasting(WeatherDemand2,DemandThreshold )
#del WeatherPrice2["Price"]
#WeatherPrice2 = WeatherPriceCombine(WeatherPrice2,Price)   # NewWeather TotalWeather
#WeatherPrice2 = pd.merge(WeatherPrice2 ,DemandSupport2,how="left",on=TimeIndex)
#
#WeatherPrice2 = WeatherPrice2.sort(TimeIndex,ascending=[1,1])
#del WeatherPrice2["WeekDay"]
#TotalTrainX, TotalTrainY, TotalTestX, AvailableDate, Tracker =  TrainAndTestGen(WeatherPrice2,PriceThreshold)
#
#
#
#
## SVM: 80- 82
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###############          Cross Validation Ticket Generation
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###############          Final Training 1
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#FunRMSE = CrossVal(AvailableDate,TotalTrainX,TotalTrainY,Param)
#Weight, BestGroup, BestMachine = QuickEnsemble(Param,FunRMSE)
#SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine,Param)
#SavingAverage2 = FinalTouch(Tracker,SaveForecast)
#FinalValue=SavingAverage1.Forecast.values
#MAE=np.absolute(FinalValue[:24]-Answer).mean()
#print FunRMSE.mean(), '  /  ', MAE, '  /  ' , (MAE+FunRMSE.mean())/2



## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###############          Evaluation
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#TotalTrainX,TotalTrainY,TotalTestX,AvailableDate,Tracker =TrainAndTestGen(WeatherPrice,PriceThreshold)
#TotalTrainX, TotalTestX = TrainingExtention(TotalTrainX, TotalTestX)
#
#FunRMSE = CrossVal(AvailableDate,TotalTrainX,TotalTrainY,Param)
#Weight, BestGroup, BestMachine = QuickEnsemble(Param,FunRMSE)
#Param['RF_NTree']=400  # 200
#SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine,Param)
#SavingAverage3 = FinalTouch(Tracker,SaveForecast)
#FinalValue=SavingAverage3.Forecast.values
#MAE=np.absolute(FinalValue[:24]-Answer).mean()
#print FunRMSE.mean(), '  /  ', MAE, '  /  ' , (MAE+FunRMSE.mean())/2
#




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Printing
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#FinalValue=(SavingAverage1["Forecast"].values+SavingAverage2["Forecast"].values)/2
#Result = PostPrinting(TargetDate, SubmissionDate, FinalValue)
#


#plt.figure()
#t=range(0,24)
#plt.plot(t,FinalValue[:24],'rs-',t,Answer,'g^-')
#plt.show()







# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#          Preparing Second Race
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# NewTrainX, NewTestsX = SecondWave(TrainOutput,SaveForecast,TotalTrainX,TotalTestX)
#FunRMSE2, SavePredict2 = CrossVal(NewTrainX,TotalTrainY)

#TrainOutput2,Weight2,BestGroup2,BestMachine2=Ensemble(FunRMSE2,SavePredict2,TotalTrainY)

#Weight2=Weight
#BestGroup2=BestGroup
#BestMachine2=BestMachine
#SaveForecast2=FinalTraining(NewTrainX,TotalTrainY,NewTestsX,Weight2, BestGroup2, BestMachine2)
#MAE=np.absolute(SaveForecast2[:24]-Answer).mean()
#print MAE

#SaveForecast2=pd.DataFrame(SaveForecast2)
#SaveForecast2.to_csv('Predictions.csv')





















#Tracker["Forecast"]=SaveForecast
#Tracker["PredictionDayOnly"]=pd.DatetimeIndex(Tracker["prediction_date"]).date
#Tracker["PredictionDayOnly"]=pd.to_datetime(Tracker["PredictionDayOnly"])
#
#FinalValue=np.zeros(MaxAhead*24)
#for i in range(0,MaxAhead):
#    A=TargetDate + datetime.timedelta(days=(i+1))
#    Sub=Tracker[Tracker.PredictionDayOnly==A]
#    Sub=Sub[["prediction_date","Forecast"]]
#    AverageForecast=pd.pivot_table(Sub,values="Forecast",index=["prediction_date"],aggfunc=np.mean)
#    FinalValue[i*24:(i+1)*24]=AverageForecast
#
#MAE=np.absolute(FinalValue[:23]-Answer).mean()
#print FunRMSE.mean(), '  /  ', MAE, '  /  ' , (MAE+FunRMSE.mean())/2









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
# Daily Average Price
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





# PTAverageGen = PTGenerationReader(PTGenerationLoc,"RealDataPT.csv")


# & (~pd.isnull(WeatherPrice["Price"]))




# TrainOutput,Weight,BestGroup,BestMachine=Ensemble(FunRMSE,SavePredict,TotalTrainY)
