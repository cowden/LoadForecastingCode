import numpy as np
import pandas as pd
import os
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import datetime

# from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


plt.rcParams['figure.figsize'] = (8.0, 4.0)
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
        Param={'CVN': 1}
        Param['MachineList']=["RF"]  # ,"GBM"
        Param['Ensem']="Simple"
        Param['GBM_NTry']=3000
        Param['GBM_LRate']=0.01
        Param['RF_NTree']=2000  # 200

    elif Selection=='Wind':
        Param={'CVN': 1}
        Param['MachineList']=["RF"] # ,","Ridge"
        Param['Ensem']="Simple"
        Param['GBM_NTry']=500
        Param['GBM_LRate']=0.2
        Param['RF_NTree']=2000  # 200

    elif Selection=='Price':
        Param={'CVN': 1}
        Param['MachineList']=["Reg"] # ,","Ridge"
        Param['Ensem']="Simple"
        Param['GBM_NTry']=500
        Param['GBM_LRate']=0.1
        Param['RF_NTree']=3000  # 200

    return Param

#######################################################################
## Price Reader
#######################################################################
def PriceReader(PriceFileName):
    RawPrice  = pd.read_csv(Loc + PriceFileName)
    RawPrice["prediction_date"]=pd.to_datetime(RawPrice["prediction_date"])
    return RawPrice


def WindLoadReader(PTGenerationFileName,LoadName,WindName):
    PTGen = pd.read_csv(Loc + PTGenerationFileName)
    PTGen["prediction_date"]=pd.to_datetime(PTGen["prediction_date"])
    PTAverageGen = pd.pivot_table(PTGen,values=["demand","Wind"],index=ID,aggfunc=np.mean)
    PTAverageGen=pd.DataFrame(PTAverageGen).reset_index()
    PTAverageGen.rename(columns={'demand': LoadName}, inplace=True)
    PTAverageGen.rename(columns={'Wind': WindName}, inplace=True)
    return PTAverageGen


def OldWeatherRead(OldWeatherFile,WindSelection,WeatherRange):
    OldWeather = pd.read_csv(OldWeatherLoc + OldWeatherFile)

    OldWeather["WindSquareA"]=OldWeather["wind_speed_100m"]**2
    OldWeather["WindSquareB"]=OldWeather["wind_speed_100m"]**3

    OldWeather["WindSquareC"]=OldWeather["wind_speed"]**2
    OldWeather["WindSquareD"]=OldWeather["wind_speed"]**3

    OldWeather["WindSquareE"]=OldWeather["wind_gust"]**2
    OldWeather["WindSquareF"]=OldWeather["wind_gust"]**3

    Single = OldWeather[ OldWeather.point==2]
    Single = Single[np.append(ID,WindSelection)]

    Twin = OldWeather[ OldWeather.point==3]
    Twin = Twin[np.append(ID,WindSelection)]

    Single = pd.merge(Single,Twin,on=ID,suffixes=('2','3'))

    for t in WeatherRange:
        Temp2 = OldWeather[ OldWeather.point==t]
        Temp2 = Temp2[np.append(ID,WindSelection)]     #  <----["temperature","radiation"]
        Single = pd.merge(Single,Temp2,on=ID,suffixes=('',str(t)))

    Single=Single[Single.Year>2014]
    Single.index=range(0,Single.shape[0])


    Single["Ahead"] = pd.DatetimeIndex(Single["prediction_date"]).hour+18

    Single["prediction_date"]=pd.to_datetime(Single["prediction_date"])
    Single["available_date"]=Single["prediction_date"] - datetime.timedelta(days=1)
    Single["available_date"]= pd.DatetimeIndex(Single["available_date"]).date
    Single["available_date"]=pd.to_datetime(Single["available_date"])

    Single["WeekDay"]=pd.DatetimeIndex(Single["prediction_date"]).weekday

    Single["DayOnly"]= pd.DatetimeIndex(Single["available_date"]).date
    Single["DayOnly"]= pd.to_datetime(Single["DayOnly"])

    return Single




def SingleWeather(WindSelection,WeatherRange):

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

        if f==0:
            Data = Single
        else:
            Data = pd.concat([Data,Single])

    Data.index=range(0,Data.shape[0])

    Data["prediction_date"]=pd.to_datetime(Data["prediction_date"])
    Data["available_date"]=pd.to_datetime(Data["available_date"])
    Temp = Data["prediction_date"] - Data["available_date"]
    Data["Ahead"]=(pd.DatetimeIndex(Temp).day-1)*24+pd.DatetimeIndex(Temp).hour

    Data["available_date"]= pd.DatetimeIndex(Data["available_date"]).date
    Data["available_date"]=pd.to_datetime(Data["available_date"])

    Data["WeekDay"]=pd.DatetimeIndex(Data["prediction_date"]).weekday
    Data["Year"]=pd.DatetimeIndex(Data["prediction_date"]).year
    Data["Month"]=pd.DatetimeIndex(Data["prediction_date"]).month
    Data["Day"]=pd.DatetimeIndex(Data["prediction_date"]).day
    Data["Hour"]=pd.DatetimeIndex(Data["prediction_date"]).hour

    Data["DayOnly"]= pd.DatetimeIndex(Data["available_date"]).date
    Data["DayOnly"]= pd.to_datetime(Data["DayOnly"])

    return Data




def TotalWeatherGen(WindSelection,WeatherRange,Indi):
    OldWeather=OldWeatherRead("historical_weather.csv",WindSelection,WeatherRange)
    NewWeather = SingleWeather(WindSelection,WeatherRange)
    if Indi==1:
        TotalWeather=pd.concat([OldWeather,NewWeather])
    else:
        TotalWeather=NewWeather

    TotalWeather = TotalWeather.sort(TimeIndex,ascending=[1,1])

    return TotalWeather



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
    Normal=(Unique - Unique.mean()) / Unique.std()

    # Set data
    TrainX=Normal.iloc[0:TrainX.shape[0],:]
    TestsX=Normal.iloc[TrainX.shape[0]:,:]

    return TrainX, TestsX



#########################################################################
## Cross Validation Function [ General Case ]
#######################################################################
def CrossVal(AvailableDate,TrainX,TrainY,Param):

    CVN=Param['CVN']
    MachineList=Param['MachineList']

    Union=np.arange(len(TrainX))

    FunRMSE=np.zeros((len(MachineList),CVN))
    for cv in range(0,CVN):
        CVTarget1 = TargetDate - datetime.timedelta(days=(CVN+5)) + datetime.timedelta(days=cv)
        CVTarget2 = CVTarget1 + datetime.timedelta(days=(5-1))
        CVTarget3 = CVTarget1 - datetime.timedelta(days=MaxAhead)
        CVTarget4 = CVTarget3 + datetime.timedelta(days=(MaxAhead-1))


        print CVTarget1,CVTarget2
        TempAvailable=AvailableDate[(AvailableDate.PredictionDayOnly>=CVTarget1)&(AvailableDate.PredictionDayOnly<=CVTarget2)]
        TempAvailable=TempAvailable[(TempAvailable.DayOnly>=CVTarget3)&(TempAvailable.DayOnly<=CVTarget4)]
        test_index=TempAvailable.index
        train_index=np.setdiff1d(Union,test_index)

#        TempAvailable=TempAvailable[TempAvailable.DayOnly==CVTarget4]
#        test_index=TempAvailable.index


        SubTrainX=TrainX.iloc[train_index,:]
        SubTrainY=TrainY.iloc[train_index]
        SubTestsX=TrainX.iloc[test_index,:]
        SubTestsY=TrainY.iloc[test_index]


        MachineValue=np.zeros((len(MachineList),5*24))
        MachineSolution=np.zeros((len(MachineList),5*24))

        for machindex in range(0,len(MachineList)):
            Machine=MachineList[machindex]

            # Outlier Detection
            SubTrainX,SubTrainY=OutlierDetector(SubTrainX,SubTrainY)

            # Preprocessing
            SubTrainX,SubTestsX=Standardization(SubTrainX, SubTestsX)

            # Forecast
            Forecast=Machinery(Machine,SubTrainX,SubTrainY,SubTestsX,Param)

            for i in range(0,5):
                SubTargetDate = CVTarget1+datetime.timedelta(days=i)
                Select=np.where( TempAvailable.PredictionDayOnly==SubTargetDate )[0]
                SubForecast=Forecast[Select]

                MA=len(SubForecast)/24
                SubWeight=np.zeros(MA)
                SubWeight[-3:]=1
                #SubWeight=range(1,MA+1)
                Weight=np.repeat(SubWeight,24)
                Weight=Weight*1.0/sum(SubWeight)
                SubForecast=SubForecast*Weight

                Answer = SubTestsY.iloc[Select]

                # For the given Prediction Date, Measure the performance
                for jj in range(0,(MaxAhead-i)):
                    SubForecast2=SubForecast[jj*24:(jj+1)*24]
                    Answer2=Answer[jj*24:(jj+1)*24]

                    TempPerformance=PerMeasure(Answer2,SubForecast2)
                    #print SubTargetDate , MaxAhead-jj-i,'   ',TempPerformance

                # For the given Prediction Date, Measure the average performance
                Temp = SubForecast.reshape(-1,24)
                Temp2 = np.sum(Temp ,axis=0)
                TempPerformance=PerMeasure(Answer[:24],Temp2)

                MachineValue[machindex,i*24:(i+1)*24] = Temp2
                MachineSolution[machindex,i*24:(i+1)*24] = Answer[:24]
                #print SubTargetDate ,'  Prediction Day  ',TempPerformance

            # Show the performance of machine
            TempPerformance=PerMeasure(MachineValue[machindex,:],MachineSolution[machindex,:])
            FunRMSE[machindex,cv]=TempPerformance
            #print Machine, '  Five Day Average  ',  TempPerformance

        SingleOutput=np.mean(MachineValue,axis=0)
        SingleSolution=np.mean(MachineSolution,axis=0)
        TempPerformance=PerMeasure(SingleOutput,SingleSolution)
        print CVTarget2, '  CV Final  ', FunRMSE[:,cv], TempPerformance
        plt.figure()
        plt.plot(range(0,len(SingleOutput)),SingleOutput,'rs-',range(0,len(SingleOutput)),SingleSolution,'b^-')
        plt.show()

    print '    '
    print np.mean(FunRMSE,axis=1)
    print '    '

    return np.mean(FunRMSE,axis=1)


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
    Selection=TrainX.Ahead < 1500
    TrainX=TrainX[Selection]
    TrainY=TrainY[Selection]

#    Reg = linear_model.LinearRegression()
#    Reg.fit(TrainX , TrainY)
#    TestOutput=Reg.predict(TrainX)
#    MAE=np.absolute(TestOutput-TrainY)
#    MAE=np.array(MAE)
#    Good=MAE.argsort()[:round(len(TrainY)*0.8)]
#    TrainX=TrainX.iloc[Good,:]
#    TrainY=TrainY.iloc[Good]
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
## Linear Regerssion
########################################################################
def LinearReg(TrainX,TrainY):
    Reg = linear_model.LinearRegression()
    return Reg




#######################################################################
## Gradient Boosting Training
#######################################################################
def GradientBoosting(TrainX,TrainY,Param):

    # Retraining
    GBM = GradientBoostingRegressor(loss='lad',n_estimators=Param['GBM_NTry'],
                                    max_depth=5,max_features=TrainX.shape[1]/3,
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
                             bootstrap=True,
                             max_features=TrainX.shape[1]/3,
                             n_jobs=-1)

    #
    # min_samples_leaf=1,
    #
    # min_samples_split
    #

    return RF




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
        GameSize=len(Param['MachineList'])
        Weight=np.ones(1)*1.0
        BestMachine=np.where(FunRMSE==np.min(FunRMSE))[0][0]
        BestMachine=np.array(BestMachine)

    elif Param['Ensem']=='Simple':
        GameSize=len(Param['MachineList'])
        Weight=np.ones(GameSize)*1.0/GameSize
        Weight=Weight[np.newaxis].transpose()
        BestMachine=range(0,len(Param['MachineList']))

    elif Param['Ensem']=='Weight':
        InverseWeight=1.0 / FunRMSE
        Weight=InverseWeight / sum(InverseWeight)
        Weight=Weight[np.newaxis].transpose()
        BestMachine=range(0,len(Param['MachineList']))

    return Weight,BestMachine






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
    return WeightedOutput







def LoadForecasting2(KownWeather,KownES,Value,Indicator,Choose,ParamIndi):
    SubWeather = pd.merge(KownWeather,KownES[np.append(ID,Value)],how="left",on=ID)
    TestXLoc = SubWeather[Value].isnull()
    DemandTestX = SubWeather[TestXLoc ]
    NextGenTest = DemandTestX[TimeIndex]
    DemandTestX = DemandTestX.drop([Value,"available_date","DayOnly","prediction_date"],1)

    DemandTrainX = SubWeather[~TestXLoc ]
    DemandTrainX.index=range(0,DemandTrainX.shape[0])
    DemandTrainY = DemandTrainX[Value]
    DemandTrainX = DemandTrainX.drop([Value,"available_date","DayOnly","prediction_date"],1)

    # Training Data Expansion
    if Indicator==1:
        DemandTrainX, DemandTestX = TrainingExtention(DemandTrainX, DemandTestX)

    ## Main Game Starts
    Param2=ParameterGen(ParamIndi)
    Weight, BestMachine = QuickEnsemble(Param2,0)
    LoadForecast=FinalTraining(DemandTrainX,DemandTrainY,DemandTestX,Weight,BestMachine,Param2)
    NextGenTest[Value]=LoadForecast
    NextGenTest = FinalTouch2(NextGenTest, Value, Choose)
    NextGenTest = NextGenTest.sort(TimeIndex,ascending=[1,1])
    SubWeather.loc[DemandTestX.index,Value]=NextGenTest[Value].values

    Output=SubWeather[["prediction_date",Value]].drop_duplicates()

    return Output





def FinalTouch2(NextGenTest, Value, Choose):
    UniqueNextGen=NextGenTest["prediction_date"].drop_duplicates()
    for i in range(0,len(UniqueNextGen)):
        TargetTime=UniqueNextGen.iloc[i]
        Sub=NextGenTest[NextGenTest.prediction_date==TargetTime]

        if Choose == "Average":
            Weight=np.ones(len(Sub))*1.0 / len(Sub)
            Weight=Weight*1.0/sum(Weight)

        elif Choose == "Best":
            SubWeight=[0]*len(Sub)
            SubWeight[-1]=1
            Weight=np.array(SubWeight)

        elif Choose == "Weight":
            SubWeight=range(1,len(Sub)+1)
            Weight=np.array(SubWeight)
            Weight=Weight*1.0/sum(Weight)

        elif Choose == "Oligar":
            SubWeight=[0]*len(Sub)
            SubWeight[-1]=1
            if len(Sub)>1:
                SubWeight[-2]=1
            elif len(Sub)>2:
                SubWeight[-3]=1
            Weight=np.array(SubWeight)
            Weight=Weight*1.0/sum(Weight)

        Sub[Value]=Sub[Value]*Weight
        Sub=Sub[["prediction_date",Value]]
        AverageForecast=pd.pivot_table(Sub,values=Value,index=["prediction_date"],aggfunc=np.sum)
        AverageForecast=pd.DataFrame(AverageForecast).reset_index()

        if i==0:
            SavingAverage=AverageForecast
        else:
            SavingAverage=pd.concat([SavingAverage,AverageForecast])

#    plt.figure()
#    plt.plot(SavingAverage[Value].values)
#    plt.show()

    NextGenTest=pd.merge(NextGenTest[TimeIndex],SavingAverage,on=["prediction_date"])
    return NextGenTest



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



#"WindSquareA","WindSquareB","WindSquareC","WindSquareD",
#


#####         Parameters
# System Parameters
SubmissionDate = "2016-04-17"
TargetDate=pd.to_datetime(SubmissionDate).date()
ID = ["prediction_date","Year","Month","Day","Hour"]
TimeIndex = ["available_date","prediction_date"]

# I/O Parameters
Loc             = "/Users/dueheelee/Documents/ComPlatt/Data/Amagedon/"
OldWeatherLoc   = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Weather/"
WeatherFileLoc  = "/Users/dueheelee/Documents/ComPlatt/Data/Forecast_Weather/"

# Weather Data
LoadRange=[1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
LoadSelection =["WindSquareA","WindSquareC","WindSquareB","WindSquareD",
                "wind_direction_100m","wind_direction",
                "temperature","radiation","wind_speed_100m","wind_speed"]

WindRange=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
WindSelection =["WindSquareA","WindSquareC","WindSquareB","WindSquareD",
                "wind_speed_100m","wind_speed"]


PriceRange=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
PriceSelection =["WindSquareA","WindSquareC","WindSquareB","WindSquareD",
                "wind_speed_100m","wind_speed"]


# "WindSquareB","WindSquareD",


Answer=np.array([19.33, 17.74, 15.85, 15.13, 15.15, 14.57, 14.8, 14.9, 16.69,18.69,
                 18.69, 18.94, 18.5,16.86, 14.89, 13.58,12.26, 14.8, 17.65, 21.19, 26.4, 42.69, 32.24,
                 27.24])



#####         Data
Price = PriceReader("NewPrice.csv")
ES = WindLoadReader("ESWind.csv","ESLoad","ESWind")
PT = WindLoadReader("PTWind.csv","PTLoad","PTWind")

## Weather
TotalWeather_Load =TotalWeatherGen(LoadSelection,LoadRange,1)
TotalWeather_Wind =TotalWeatherGen(WindSelection,WindRange,1)
TotalWeather_Price=TotalWeatherGen(PriceSelection,PriceRange,1)

##### CV
CVN=1
WindEffect=10
Choose="Average"  # Oligar
ESWindMAE=0
PTWindMAE=0

FunRMSE=np.zeros(CVN)
for cv in range(0,CVN):

    if CVN==1:
        CVTarget =TargetDate+ datetime.timedelta(days=1)
    else:
        CVTarget = TargetDate - datetime.timedelta(days=(CVN+5)) + datetime.timedelta(days=cv)

    CVTargetBound = CVTarget + datetime.timedelta(days=5)





###### Data Gen
    KnownPrice = Price[Price.prediction_date<CVTarget]
    KownES = ES[ES.prediction_date < CVTarget]
    KownPT = PT[PT.prediction_date < CVTarget]

    KownWeather_Load = TotalWeather_Load[TotalWeather_Load.prediction_date < CVTargetBound]
    KownWeather_Load = KownWeather_Load[KownWeather_Load.available_date < CVTarget]

    KownWeather_Wind= TotalWeather_Wind[TotalWeather_Wind.prediction_date < CVTargetBound]
    KownWeather_Wind = KownWeather_Wind[KownWeather_Wind.available_date < CVTarget]

    KownWeather_Price= TotalWeather_Price[TotalWeather_Price.prediction_date < CVTargetBound]
    KownWeather_Price = KownWeather_Price[KownWeather_Price.available_date < CVTarget]







#### ES Load Forecasting
#    ESLoad = LoadForecasting2(KownWeather_Load,KownES,"ESLoad",0,Choose,'Load')
#    ESLoad.to_csv('ESLoad.csv',index_label=None,index=False)
    ESLoad = pd.read_csv('ESLoad.csv')
    ESLoad["prediction_date"]=pd.to_datetime(ESLoad["prediction_date"])

    Forecast = pd.merge(ESLoad.iloc[-121:],ES[["prediction_date","ESLoad"]],how="left",on="prediction_date")
    ESLoadMAE = PerMeasure(Forecast["ESLoad_x"],Forecast["ESLoad_y"])
    t=np.arange(0,len(Forecast["ESLoad_x"]))
    plt.figure()
    plt.title('ESLoad  ' + str(ESLoadMAE))
    plt.plot(t,Forecast["ESLoad_x"],'rs-',t,Forecast["ESLoad_y"],'b^-')
    plt.show()


#### PT LoadGen
#    PTLoad = LoadForecasting2(KownWeather_Load,KownPT,"PTLoad",0,Choose,'Load')
#    PTLoad.to_csv('PTLoad.csv',index_label=None,index=False)
    PTLoad = pd.read_csv('PTLoad.csv')
    PTLoad["prediction_date"]=pd.to_datetime(PTLoad["prediction_date"])

    Forecast = pd.merge(PTLoad.iloc[-121:],PT[["prediction_date","PTLoad"]],how="left",on="prediction_date")
    PTLoadMAE = PerMeasure(Forecast["PTLoad_x"],Forecast["PTLoad_y"])
    plt.figure()
    plt.title('PTLoad   ' + str(PTLoadMAE))
    plt.plot(t,Forecast["PTLoad_x"],'rs-',t,Forecast["PTLoad_y"],'b^-')
    plt.show()









####### Merging
    WithLoad = pd.merge(KownWeather_Price,ESLoad,how="left",on="prediction_date")
    WithLoad = pd.merge(WithLoad   ,PTLoad,how="left",on="prediction_date")


####### Wind Effect
    if WindEffect != 121:
        ESWind = LoadForecasting2(KownWeather_Wind,KownES,"ESWind",0,Choose,'Wind')
        Forecast = pd.merge(ESWind.iloc[-121:],ES[["prediction_date","ESWind"]],how="left",on="prediction_date")
        ESWindMAE = PerMeasure(Forecast["ESWind_x"],Forecast["ESWind_y"])
        t=np.arange(0,len(Forecast["ESWind_x"]))
        plt.figure()
        plt.title('ESWind  ' + str(ESWindMAE))
        plt.plot(t,Forecast["ESWind_x"],'rs-',t,Forecast["ESWind_y"],'b^-')
        plt.show()




        PTWind = LoadForecasting2(KownWeather_Wind,KownPT,"PTWind",0,Choose,'Wind')
        Forecast = pd.merge(PTWind.iloc[-121:],PT[["prediction_date","PTWind"]],how="left",on="prediction_date")
        PTWindMAE = PerMeasure(Forecast["PTWind_x"],Forecast["PTWind_y"])
        plt.figure()
        plt.title('PTWind  ' + str(PTWindMAE) )
        plt.plot(t,Forecast["PTWind_x"],'rs-',t,Forecast["PTWind_y"],'b^-')
        plt.show()



        WithWind = pd.merge(WithLoad, ESWind,how="left",on="prediction_date")
        WithWind = pd.merge(WithWind, PTWind,how="left",on="prediction_date")

        Price_Wind = LoadForecasting2(WithWind, KnownPrice,"Price",0,Choose,'Price')
        Price_Wind_Fortion = Price_Wind.iloc[-121:-WindEffect,]

        Price_Load = LoadForecasting2(WithLoad,KnownPrice,"Price",0,Choose,'Price')
        Price_Load_Fortion = Price_Load.iloc[-WindEffect:,]
        Forecast = pd.concat([Price_Wind_Fortion,Price_Load_Fortion])

    elif WindEffect == 121:
        Price_Load = LoadForecasting2(WithLoad,KnownPrice,"Price",0,"Oligar",'Price')
        Forecast = Price_Load.iloc[-WindEffect:,]









    ## Price Forecasting

    # Final Evaluation or Printing
    if CVN > 1:
        Forecast = pd.merge(Forecast,Price[["prediction_date","Price"]],how="left",on="prediction_date")
        MAE = PerMeasure(Forecast["Price_x"],Forecast["Price_y"])

        t=np.arange(0,len(Forecast["Price_y"]))
        plt.figure()
        plt.title('Price  ' + str(MAE))
        plt.plot(t,Forecast["Price_x"],'rs-',t,Forecast["Price_y"],'b^-')
        plt.show()


        FunRMSE[cv]=MAE
        print CVTarget, CVTargetBound, '  ' ,ESLoadMAE, PTLoadMAE, ESWindMAE, PTWindMAE, MAE

    else:
#        MAE = PerMeasure(Forecast.Price[:24] , Answer)
#        print MAE
#
#        t=np.arange(0,len(Answer))
#        plt.figure()
#        plt.plot(t,Forecast.Price[:24],'rs-',t,Answer,'b^-')
#        plt.show()

        Result = PostPrinting(TargetDate, SubmissionDate, Forecast.Price.values)











print FunRMSE.mean()









