


#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def ParameterSetting():
    # Cross Validation Parameters
    Param={'CVN': 5, 'CVOptExternal':"HoldOut"}

    Param['CVMethod']=['KFold']
    Param['SubCVN']=5

    # Machine List
    Param['MachineList']=['RF','Ridge']   #  RF GBM XGB

    # Group Number
    Param['NGroup']=[1,6,7]  # ,3,4,6

    # Ensem MEthod
    Param['Ensem']="Weight"

    # GBM
    Param['GBM_NTry']=2000  # 4000
    Param['GBM_LRate']=0.07
    Param['GBM_CVP']=0.1

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

