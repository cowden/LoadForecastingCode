import numpy as np
import pandas as pd


#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def ParameterSetting():
    # Cross Validation Parameters
    Param={'CVN': 5, 'CVOptExternal':"HoldOut"}

    Param['CVMethod']=['KFold']
    Param['SubCVN']=5

    # Machine List
    Param['MachineList']=['GBM','Ridge']   #  RF GBM XGB

    # Group Number
    Param['NGroup']=[1,6,7]  # ,3,4,6

    # Ensem MEthod
    Param['Ensem']="Weight"

    # GBM
    Param['GBM_NTry']=[100,200,300]  # 4000
    Param['GBM_LRate']=[0.1]
    Param['GBM_SubSample']=[0.5]
    Param['MaxDepth']=[5]
    Param['GBM_Split']=[2]
    Param['GBM_Leaf']=[1]
    Param['Loss']='ls'
    Param['Alpha']=0.5

    # Random Forest
    Param['RF_CVN']=10
    Param['RF_NTree']=[100]
    Param['RF_Feature']=["auto"]  #['auto', 'sqrt', 'log2']
    Param['RF_Split']=[2,3,4]
    Param['RF_Leaf']=[1]

    # SVM
    Param['SVM_CVP']=0.1
    Param['C']=[0.1,10,100,300,500,1000,10000,100000]
    Param['Gamma']=0.0

    # KNN
    Param['NNeighbors']=10

    # Ridge Regression
    Param['Reg_Reg']=[50,100,150,200,150,300,350]

    # Min and Max for preprocessing
    Param['Min']=0
    Param['Max']=10


    return Param



#######################################################################
## Outlier Detector  [ Specify ]
#######################################################################
def OutlierDetector(TrainX,TrainY):

    CleanID=TrainX.isnull().any(axis=1)

    TrainX2=TrainX[~CleanID]
    TrainY2=TrainY[~CleanID]

    # Outlier Detection
    return TrainX2, TrainY2




#######################################################################
## Group Gen[ Specify ]
############################################################################
def GroupGen(TrainX, TestX, GroupDecision):
    TotalX=pd.concat([TrainX , TestX])
    TotalX.index=range(0,TotalX.shape[0])

    if GroupDecision=="Hour":
        NumericRange=range(0,24)
        GroupRange=["D"+str(i) for i in map(str,NumericRange)]

        Dummy=pd.DataFrame(index=range(0,TotalX.shape[0]),columns=GroupRange)
        Dummy=Dummy.fillna(0)

        # Generate a dummy variable matrix
        for t in NumericRange:
            LocHour=np.where(TotalX.Hour==NumericRange[t])[0]
            Dummy.iloc[LocHour,t]=1

        TotalX2=pd.concat([TotalX,Dummy],axis=1)

    elif GroupDecision=="Quantile":
        GroupRange=range(0,2)
        #Windquantile=TotalX.U10_1.quantile([.5])

    elif GroupDecision=="None":
        TotalX2=TotalX.copy()

    TrainX2=TotalX2.iloc[0:TrainX.shape[0],:]
    TestX2=TotalX2.iloc[0:TestX.shape[0],:]

    return TrainX2, TestX2
############################################################################


############################################################################
def TimeInfo(DataFrame):
    DataFrame["Hour"] = pd.DatetimeIndex(DataFrame["TIMESTAMP"]).hour
    DataFrame["Day"] = pd.DatetimeIndex(DataFrame["TIMESTAMP"]).day
    DataFrame["Month"] = pd.DatetimeIndex(DataFrame["TIMESTAMP"]).month
    DataFrame["Year"] = pd.DatetimeIndex(DataFrame["TIMESTAMP"]).year
    DataFrame["WeekDay"] = pd.DatetimeIndex(DataFrame["TIMESTAMP"]).weekday

    return DataFrame
############################################################################





def FeatureEng(TrainX, TestX):
    TrainX=TimeInfo(TrainX)
    TestX=TimeInfo(TestX)

    #TIMESTAMP=TrainX["TIMESTAMP"]
    del TrainX["TIMESTAMP"]
    del TestX["TIMESTAMP"]

    return TrainX, TestX










