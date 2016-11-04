import numpy as np
import pandas as pd
import os
#import datetime


# Options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', 50000)



#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def ParameterSetting(NClass):
	# Cross Validation Parameters
	Param={'CVN': 5, 'CVP':0.1, 'CVOptExternal':"HoldOut"}

	# Machine List
	Param['MachineList']=['LogReg','RF']   # LogReg KNN GBM

	# Group Number
	Param['NGroup']=[2,3]

	# Ensemble Option
	Param['Ensem']="Weight"

	# Internal Cross Validation Option
	Param['CVOptInternal']="HoldOut"

	# GBM
	Param['GBM_NTry']=500
	Param['GBM_LRate']=0.02
	Param['GBM_CVP']=0.1

	# Random Forest
	Param['RF_NTree']=10
	Param['RF_CVP']=0.1

	# SVM
	Param['SVM_CVP']=0.1
	Param['C']=[3,5,7,8,9,10]
	Param['Gamma']=0.5

	# NuSVM
	Param['SVM_CVP']=0.1
	Param['Nu']=[0.0001, 0.0002, 0.0003, 0.0005, 0.001]

	# KNN
	Param['NNeighbors']=3000

	# Logistic Regression
	Param['Log_Reg']=[200] #[0.1,0.5,1,5,10,30,50,100,300,500,700,900,1200,1500,2000]
	Param['Log_CVN']=1
	Param['Log_CVP']=0.1

	# Number of Class
	Param['NC']=NClass
	Param['Min']=1e-4
	Param['Max']=1

	# XGBoost Parameters
	params = {}
	params["objective"] = "reg:linear"
	params["eta"] = 0.05
	params["min_child_weight"] = 5
	params["subsample"] = 0.8
	params["colsample_bytree"] = 0.8
	params["scale_pos_weight"] = 1.0
	params["silent"] = 1
	params["max_depth"] = 9
	plst = list(params.items())
	Param['XGB']=plst
	Param['XGB_NR']=300   # 300


	return Param









###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

##### Price Information ###############
WeatherFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Historical_Price/"
RawPrice  = pd.read_csv(WeatherFileLoc + "NewPrice.csv")
RawPrice["Date"]=pd.to_datetime(RawPrice["Date"])
RawPrice .rename(columns={'Date': 'prediction_date'},inplace=True)

RawPrice=RawPrice.loc[:RawPrice.shape[0]-25,:]

###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


TestCol=["available_date","prediction_date","Price"]


##############          Forecated Daily Weahter Data Collection         ##############

## Read Data First
WeatherFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Forecast_Weather/"
FileList=os.listdir(WeatherFileLoc)
FileList.sort()


##
for f in range(0,len(FileList)):
    ReadData  = pd.read_csv(WeatherFileLoc + FileList[f])
    ReadData["available_date"]=pd.to_datetime(ReadData["available_date"])
    ReadData["prediction_date"]=pd.to_datetime(ReadData["prediction_date"])

    if f==0:
        Data = ReadData
    else:
        Data = pd.concat([Data,ReadData])

##
Data.index=range(0,Data.shape[0])


##
TargetDate=ReadData.loc[0,"available_date"]




##
ComWeather = Data[ Data.point==1]
del ComWeather["point"]
Temp2 = Data[Data.point==2]
del Temp2["point"]
ComWeather = pd.merge(ComWeather,Temp2,on=["available_date","prediction_date"],suffixes=('1','2'))
for t in range(3,19):
    Temp2 = Data[Data.point==t]
    del Temp2["point"]
    ComWeather = pd.merge(ComWeather,Temp2,on=["available_date","prediction_date"],suffixes=('',str(t)))

##
ComWeather .rename(columns={'wind_speed_100m': 'wind_speed_100m3',
                            'wind_direction_100m': 'wind_direction_100m3',
                            'temperature': 'temperature3',
                            'air_density': 'air_density3',
                            'pressure': 'pressure3',
                            'precipitation': 'precipitation3',
                            'wind_gust': 'wind_gust3',
                            'radiation': 'radiation3',
                            'wind_speed': 'wind_speed3',
                            'wind_direction': 'wind_direction3'}, inplace=True)


##
WeatherPrice=pd.merge(ComWeather,RawPrice,how="left",on="prediction_date")
ColNames=list(WeatherPrice.columns.values)
NewColName=ColNames[-5:]+ColNames[:-5]
WeatherPrice=WeatherPrice[NewColName]



TotalTest=WeatherPrice[WeatherPrice.available_date==TargetDate];
TotalTrain=WeatherPrice[(WeatherPrice.available_date!=TargetDate) & (~np.isnan(WeatherPrice.Price))];













## Grouping
IndexList = np.array(range(0,WeatherPrice.shape[0])).reshape(1,-1)
IndexList = np.reshape(IndexList,(len(FileList),7,24))


##
Transfer = [0] * 7
for h in range(0,7):
    Selection = IndexList[:,h,:].reshape(1,-1)
    Selection = Selection.reshape(-1)
    Transfer[h]=WeatherPrice.iloc[Selection,:]




###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$








# A=WeatherPrice[["Price","available_date","prediction_date"]]




















# Current directory
# os.getcwd()
#
#ReadData["Year"]=ReadData["prediction_date"].map(lambda x: x.year)
#ReadData["Month"]=ReadData["prediction_date"].map(lambda x: x.month)
#ReadData["Day"]=ReadData["prediction_date"].map(lambda x: x.day)
#ReadData["Hour"]=ReadData["prediction_date"].map(lambda x: x.hour)
#del ReadData["prediction_date"]
## ReadData["A"]=pd.DatetimeIndex(ReadData["prediction_date"]).year








#    Template = pd.merge(Outputs_ECM[ColNames],Final7[ColNames],on=SixTag,suffixes=("_ECM","_MY"))








##A=datetime.datetime.strptime(Data.prediction_date[1:2], "%Y-%m-%d %H:%M:%S")


# Final45=pd.concat([SilverCata_AreaState , NoSilverNoCata_AreaState])




# A=datetime.strptime(Temp.available_date[1],"%Y-%m-%d %H:%M:%S")




#FileList = os.listdir(WeatherFileLoc + TargetFileName)





#Original = pd.read_csv(Folder+FileName)			# Read a file



