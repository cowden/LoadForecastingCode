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
        Param={'CVN': 1}
        Param['MachineList']=["Reg"]  # ,"GBM"
        Param['Ensem']="Simple"
        Param['GBM_NTry']=3000
        Param['GBM_LRate']=0.01
        Param['RF_NTree']=3000  # 200

    elif Selection=='Price':
        Param={'CVN': 10}
        Param['MachineList']=["Reg"] # ,","Ridge"
        Param['Ensem']="Simple"
        Param['GBM_NTry']=3000
        Param['GBM_LRate']=0.01
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


def OldWeatherRead(OldWeatherFile,WindSelection):
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




def SingleWeather(WindSelection):

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
## Gradient Boosting Training
#######################################################################
def GradientBoosting(TrainX,TrainY,Param):

    # Retraining
    GBM = GradientBoostingRegressor(loss='lad',n_estimators=Param['GBM_NTry'],
                                    max_depth=5,max_features=TrainX.shape[1]/2,
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


def LoadForecasting2(KownWeather,KownES,Value,Indicator,Choose):

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
    Param2=ParameterGen('Load')
    Weight, BestMachine = QuickEnsemble(Param2,0)
    LoadForecast=FinalTraining(DemandTrainX,DemandTrainY,DemandTestX,Weight,BestMachine,Param2)
    NextGenTest[Value]=LoadForecast
    NextGenTest = FinalTouch2(NextGenTest, Value, Choose)
    NextGenTest = NextGenTest.sort(TimeIndex,ascending=[1,1])
    SubWeather.loc[DemandTestX.index,Value]=NextGenTest[Value].values

    Output=SubWeather[["prediction_date",Value]].drop_duplicates()

    return Output






# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##         Load Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def LoadForecasting(WeatherDemand,Indicator):


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
    FunRMSE=CrossVal(AvailableDate,DemandTrainX,DemandTrainY,Param2)

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




def FinalTouch2(NextGenTest, Value, Choose):
    UniqueNextGen=NextGenTest["prediction_date"].drop_duplicates()
    for i in range(0,len(UniqueNextGen)):
        TargetTime=UniqueNextGen.iloc[i]
        Sub=NextGenTest[NextGenTest.prediction_date==TargetTime]

        if Choose == "Average":
            Weight=np.ones(len(Sub))*1.0 / len(Sub)
            Weight=Weight*1.0/sum(Weight)

        elif Choose == "Best":
            SubWeight=range(1,len(Sub)+1)
            Weight=np.array(SubWeight)
            Weight=Weight*1.0/sum(Weight)
            Sub[Value]=Sub[Value]*Weight

        elif Choose == "Weight":
            SubWeight=range(1,len(Sub)+1)
            Weight=np.array(SubWeight)
            Weight=Weight*1.0/sum(Weight)
            Sub[Value]=Sub[Value]*Weight

        elif Choose == "Oligar":
            SubWeight=[0]*len(Sub)
            SubWeight[-1]=1
            SubWeight[-2]=1
            SubWeight[-3]=1
            Weight=np.array(SubWeight)

        Sub[Value]=Sub[Value]*Weight
        Sub=Sub[["prediction_date",Value]]
        AverageForecast=pd.pivot_table(Sub,values=Value,index=["prediction_date"],aggfunc=np.sum)
        AverageForecast=pd.DataFrame(AverageForecast).reset_index()

        if i==0:
            SavingAverage=AverageForecast
        else:
            SavingAverage=pd.concat([SavingAverage,AverageForecast])

    plt.figure()
    plt.plot(SavingAverage[Value].values)
    plt.show()

    NextGenTest=pd.merge(NextGenTest[TimeIndex],SavingAverage,on=["prediction_date"])
    return NextGenTest

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##         Final Result Averaging
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def FinalTouch(Tracker,SaveForecast):
    Tracker["Forecast"]=SaveForecast
    UniqueNextGen=Tracker["prediction_date"].drop_duplicates()

    for i in range(0,121):  # len(UniqueNextGen)
        TargetTime=UniqueNextGen.iloc[i]
        Sub=Tracker[Tracker.prediction_date==TargetTime]


        # Weight
#        Weight=np.ones(len(Sub))
#        SubWeight=[0]*len(Sub)
#        SubWeight[-1]=1
#        SubWeight[-2]=1
#        SubWeight[-3]=1
        SubWeight=range(1,len(Sub)+1)
        Weight=np.array(SubWeight)
        Weight=Weight*1.0/sum(Weight)
        Sub.Forecast=Sub.Forecast*Weight

        # First
        #Sub=Sub[Sub.available_date==Sub.available_date.max()]
        Sub=Sub[["prediction_date","Forecast"]]
        AverageForecast=pd.pivot_table(Sub,values="Forecast",
                                       index=["prediction_date"],aggfunc=np.sum)
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



#"WindSquareA","WindSquareB","WindSquareC","WindSquareD",
#"temperature","radiation","precipitation","air_density","pressure"


#####         Parameters
# System Parameters
SubmissionDate = "2016-04-15"
TargetDate=pd.to_datetime(SubmissionDate).date()
ID = ["prediction_date","Year","Month","Day","Hour"]
TimeIndex = ["available_date","prediction_date"]

# I/O Parameters
Loc             = "/Users/dueheelee/Documents/ComPlatt/Data/Amagedon/"
OldWeatherLoc   = "/Users/dueheelee/Documents/ComPlatt/Data/Historical_Weather/"
WeatherFileLoc  = "/Users/dueheelee/Documents/ComPlatt/Data/Forecast_Weather/"

# Weather Data
WeatherRange=[4] # ,5,6,7,8,9,10,11,12,13,14,15,16,17
WindSelection =["wind_speed_100m","wind_speed","wind_gust"]






#####         Data
Price = PriceReader("NewPrice.csv")
ES = WindLoadReader("ESWind.csv","ESLoad","ESWind")
PT = WindLoadReader("PTWind.csv","PTLoad","PTWind")

## Forecasted Weather Data
OldWeather=OldWeatherRead("historical_weather.csv",WindSelection)
NewWeather = SingleWeather(WindSelection)
TotalWeather=pd.concat([OldWeather,NewWeather])
TotalWeather = TotalWeather.sort(TimeIndex,ascending=[1,1])








##### Simulation
CVN=5
cv=1
WindIndicator=0
CVTarget = TargetDate - datetime.timedelta(days=(CVN+6)) + datetime.timedelta(days=cv)
CVTargetBound = CVTarget + datetime.timedelta(days=5)

# Data Gen
KnownPrice = Price[Price.prediction_date<CVTarget]
KownES = ES[ES.prediction_date < CVTarget]
KownPT = PT[PT.prediction_date < CVTarget]
KownWeather = TotalWeather[TotalWeather.prediction_date < CVTargetBound]
KownWeather = KownWeather[KownWeather.available_date < CVTarget]


# ES Load Forecasting
Choose="Average"
ESLoad = LoadForecasting2(KownWeather,KownES,"ESLoad",0,Choose)
PTLoad = LoadForecasting2(KownWeather,KownPT,"PTLoad",0,Choose)
ESWind = LoadForecasting2(KownWeather,KownES,"ESWind",0,Choose)
PTWind = LoadForecasting2(KownWeather,KownPT,"PTWind",0,Choose)


## Merging
KownWeather = pd.merge(KownWeather,ESLoad,how="left",on="prediction_date")
KownWeather = pd.merge(KownWeather,PTLoad,how="left",on="prediction_date")

if WindIndicator==1:
    KownWeather=pd.merge(KownWeather,ESWind,how="left",on="prediction_date")
    KownWeather=pd.merge(KownWeather,PTWind,how="left",on="prediction_date")


## Price Forecasting
PriceForecast = LoadForecasting2(KownWeather,KnownPrice,"Price",0,Choose)
Forecast = PriceForecast.iloc[-120:,]
Forecast = pd.merge(Forecast,Price[["prediction_date","Price"]],how="left",on="prediction_date")
MAE = PerMeasure(Forecast["Price_x"],Forecast["Price_y"])











# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Wind Power Forecasting
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Fast=0
if Fast==1:
    DemandSupport = pd.read_csv('Load.csv')
    DemandSupport["prediction_date"]=pd.to_datetime(DemandSupport["prediction_date"])
    DemandSupport["available_date"]=pd.to_datetime(DemandSupport["available_date"])

    ESWindSupport = pd.read_csv('W1.csv')
    ESWindSupport["prediction_date"]=pd.to_datetime(ESWindSupport["prediction_date"])
    ESWindSupport["available_date"]=pd.to_datetime(ESWindSupport["available_date"])

#    PTWindSupport = pd.read_csv('W2.csv')
#    PTWindSupport["prediction_date"]=pd.to_datetime(PTWindSupport["prediction_date"])
#    PTWindSupport["available_date"]=pd.to_datetime(PTWindSupport["available_date"])

else:

#    WeatherRange=[1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #1 5 9 13 17
#    WindSelection =["temperature","pressure","wind_speed_100m","wind_speed","wind_gust","precipitation","radiation"]
#    SubSelection=WindSelection
#    OldWeather=OldWeatherRead(OldWeatherLoc, "historical_weather.csv",WindSelection,SubSelection)
#    NewWeather = SingleWeather(WeatherFileLoc,WindSelection,SubSelection)
#    TotalWeather=pd.concat([OldWeather,NewWeather])


    WeatherRange=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    WindSelection =["wind_speed_100m","wind_speed"]
    SubSelection=WindSelection
    OldWeather = OldWeatherRead(OldWeatherLoc, "historical_weather.csv",WindSelection,SubSelection)
    NewWeather = SingleWeather(WeatherFileLoc,WindSelection,SubSelection)
    TotalWeather=pd.concat([OldWeather,NewWeather])

    DemandSupport = pd.read_csv('Load.csv')
    DemandSupport["prediction_date"]=pd.to_datetime(DemandSupport["prediction_date"])
    DemandSupport["available_date"]=pd.to_datetime(DemandSupport["available_date"])



    # Load Forecasting
#    SubWeather=pd.merge(TotalWeather,Energy,how="left",on=ID)
#    SubWeather = pd.merge(SubWeather,PTAverageGen,how="left",on=ID)
#    DemandSupport = LoadForecasting(SubWeather,DemandThreshold,0)
#    DemandSupport.to_csv('Load.csv',index_label=None,index=False)

    # ES Wind Forecasting
    SubWeather=pd.merge(TotalWeather,ESWind,how="left",on=ID)
    SubWeather.rename(columns={'Wind': 'Demand'}, inplace=True)
    ESWindSupport = LoadForecasting(SubWeather,ESWindThreshold,0)
    ESWindSupport.rename(columns={'Demand': 'ESWind'}, inplace=True)
    ESWindSupport.to_csv('W1.csv',index_label=None,index=False)

#    # PT Wind Forecasting
#    SubWeather=pd.merge(TotalWeather,PTWind,how="left",on=ID)
#    SubWeather.rename(columns={'Wind': 'Demand'}, inplace=True)
#    PTWindSupport = LoadForecasting(SubWeather,PTWindThreshold,0)
#    PTWindSupport.rename(columns={'Demand': 'PTWind'}, inplace=True)
#    PTWindSupport.to_csv('W2.csv',index_label=None,index=False)



#"WindSquareD",

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherRange=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
WindSelection =["WindSquareA","WindSquareC","WindSquareB","WindSquareD","wind_speed_100m","wind_speed"]
SubSelection=WindSelection #["temperature","radiation"]
OldWeather=OldWeatherRead(OldWeatherLoc, "historical_weather.csv",WindSelection,SubSelection)
NewWeather = SingleWeather(WeatherFileLoc,WindSelection,SubSelection)
TotalWeather=pd.concat([OldWeather,NewWeather])



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##############          Make Weather + Price Data
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
WeatherPrice=WeatherPriceCombine(TotalWeather,Price)   # NewWeather TotalWeather
WeatherPrice=pd.merge(WeatherPrice ,DemandSupport,how="left",on=TimeIndex)
WeatherPrice=pd.merge(WeatherPrice ,Energy[["prediction_date","DemandPT"]],how="left",on=["prediction_date"])
WeatherPrice=pd.merge(WeatherPrice ,ESWindSupport,how="left",on=TimeIndex)
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
#FunRMSE =0
FunRMSE = CrossVal(AvailableDate,TotalTrainX,TotalTrainY,Param)
Weight, BestGroup, BestMachine = QuickEnsemble(Param,FunRMSE)
SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine,Param)
SavingAverage1 = FinalTouch(Tracker,SaveForecast)
#FinalValue=SavingAverage1.Forecast.values
#MAE=np.absolute(FinalValue[:24]-Answer).mean()
#print FunRMSE.mean() , '  /  ', MAE, '  /  ' , np.append(FunRMSE,MAE).mean()
#
#
#t=range(0,24)
#plt.plot(t,FinalValue[:24],'rs-',t,Answer,'b^-')
#plt.show()








## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###############          LoadForecasting Revisit
## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#WeatherRange=[1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#WindSelection =["WindSquareA","WindSquareB","WindSquareC","WindSquareD",
#                "temperature","radiation","wind_gust","precipitation",
#                "wind_speed_100m","wind_speed"]
#SubSelection=WindSelection
#OldWeather=OldWeatherRead(OldWeatherLoc, "historical_weather.csv",WindSelection,SubSelection)
#NewWeather = SingleWeather(WeatherFileLoc,WindSelection,SubSelection)
#TotalWeather=pd.concat([OldWeather,NewWeather])
#
#WeatherPrice=WeatherPriceCombine(TotalWeather,Price)   # NewWeather TotalWeather
#WeatherPrice=pd.merge(WeatherPrice ,DemandSupport,how="left",on=TimeIndex)
#WeatherPrice=pd.merge(WeatherPrice ,Energy[["prediction_date","DemandPT"]],how="left",on=["prediction_date"])
##WeatherPrice=pd.merge(WeatherPrice ,ESWindSupport,how="left",on=TimeIndex)
##WeatherPrice=pd.merge(WeatherPrice ,PTWindSupport,how="left",on=TimeIndex)
#
#WeatherPrice = WeatherPrice.sort(TimeIndex,ascending=[1,1])
##del WeatherPrice["Day"]
##del WeatherPrice["Year"]
##del WeatherPrice["WeekDay"]    ## Always Delete
##del WeatherPrice["Month"]    ## Always Delete
#
#TotalTrainX, TotalTrainY, TotalTestX, AvailableDate, Tracker = TrainAndTestGen(WeatherPrice,PriceThreshold)
#
#Param=ParameterGen('Price')
#Param['MachineList']=["RF","GBM"]
#FunRMSE = CrossVal(AvailableDate,TotalTrainX,TotalTrainY,Param)
#Weight, BestGroup, BestMachine = QuickEnsemble(Param,FunRMSE)
#SaveForecast=FinalTraining(TotalTrainX,TotalTrainY,TotalTestX,Weight,BestGroup,BestMachine,Param)
#SavingAverage1 = FinalTouch(Tracker,SaveForecast)
#
#
#
#
#t=range(0,24)
#plt.plot(t,FinalValue[:24],'rs-',t,Answer,'b^-')
#plt.show()





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
#FinalValue=SavingAverage1["Forecast"].values
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







#        Single["Ahead"] = np.repeat(range(1,8),24)
#        #Single=Single[Single["Ahead"]<6]
