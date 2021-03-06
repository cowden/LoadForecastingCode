# For All Forecastings


import numpy as np
import pandas as pd
import os
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import datetime

# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor




### My Own Code
import CasePrep.GEF2014_Helper as Helper
import ForecastingSkills.Regression3 as Regression



### Homework
# 1. Visualization
# 2. Program transfer between point and probabilistic forecasting
# 3. Implement the neural network
# 4. Implement the deep learning
# 5. Implement the outlier detection
# 6. Implement the column selection
# 7. Implement the GBM and RF Parameter Optimization technique





###########       GEFCom2014 depending code            ###########
## Important Parameters

TaskNumber = 14
DataCenter = "/Users/dueheelee/Documents/Drop/WindForecasting/Data/GEFCom2014Wind/"
GroupDecision="None"   # "None" "Hour"
############################################################################



############################################################################
def InstantZonalPrep(RawData, Zone):

    DF=RawData[["TIMESTAMP","U10","V10","U100","V100"]]

    DF.loc[:,"SquareU10"]=DF["U10"]**2
    DF.loc[:,"SquareV10"]=DF["V10"]**2

    DF.loc[:,"SquareU100"]=DF["U100"]**2
    DF.loc[:,"SquareV100"]=DF["V100"]**2

    DF.loc[:,"CubicU10"]=DF["U10"]**3
    DF.loc[:,"CubicV10"]=DF["V10"]**3

    DF.loc[:,"CubicU100"]=DF["U100"]**3
    DF.loc[:,"CubicV100"]=DF["V100"]**3

    TempColNames= DF.columns[1:] + '_' + str(Zone)
    TempColNames = np.append("TIMESTAMP", TempColNames.values)
    DF.columns=TempColNames

    return DF
############################################################################








############################################################################
## Wind Data Reader
for i in range(0,1):  # 10

    ## Determine Zone Number
    Zone=i+1
    print(Zone)

    ## Read Past Training Data
    TrainFileName="Task" + str(TaskNumber) + "_W_Zone" + str(Zone)+ ".csv"
    TrainRawData=pd.read_csv(DataCenter + TrainFileName)


    ## Split data into wind power and wind speed
    PowerData=TrainRawData[["ZONEID","TIMESTAMP","TARGETVAR"]]

    ## Data Manipulation ( Variable Engineering )
    TrainSpeed=InstantZonalPrep(TrainRawData, Zone)


    ## Read Future Weather Data
    TestFileName="TaskExpVars" + str(TaskNumber) + "_W_Zone" + str(i+1)+ ".csv"
    TestRawData=pd.read_csv(DataCenter + TestFileName)


    ## Data Manipulation ( Variable Engineering )
    TestSpeed=InstantZonalPrep(TestRawData, Zone)


    ## Data Combination
    if i==0:
        TrainX=TrainSpeed
        TrainY=PowerData
        TestX=TestSpeed

    else:
        TrainX=pd.merge(TrainX,TrainSpeed,on='TIMESTAMP',how='outer')
        TrainY=pd.concat([TrainY,PowerData])
        TestX=pd.merge(TestX,TestSpeed,on='TIMESTAMP',how='outer')
############################################################################





## 2. Parameter Settings
Param=Helper.ParameterSetting()



## 3. Post Data Engineering Across all training and testing
TrainX,TestX=Helper.FeatureEng(TrainX,TestX)



## 4. Grouping with Dommny Variable
TrainX,TestX=Helper.GroupGen(TrainX, TestX, GroupDecision)



## 5. Game Start
for z in range(1,2):

    SubTrainY=TrainY[TrainY.ZONEID==z].TARGETVAR
    MachineRMSE, SavePredict = Regression.CrossVal(Param,TrainX,SubTrainY)
    print "Zone:",z, "Performance:",MachineRMSE
















## Performance Testing and Select Best One
## Cross Validation








## 1st Direct Point Forecasting













## Direct Probabilistic Forecasting


































