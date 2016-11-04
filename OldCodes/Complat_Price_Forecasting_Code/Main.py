import numpy as np
import pandas as pd
import os
import datetime


# Options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', 500)



# Current directory
# os.getcwd()




##############          Forecated Daily Weahter Data Collection         ##############

## Read Data First
WeatherFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Forecast_Weather/"
FileList=os.listdir(WeatherFileLoc)
FileList.sort()


## ReadData["A"]=pd.DatetimeIndex(ReadData["prediction_date"]).year
for f in range(0,len(FileList)):
    ReadData  = pd.read_csv(WeatherFileLoc + FileList[f])
    ReadData["prediction_date"]=pd.to_datetime(ReadData["prediction_date"])
    ReadData["Year"]=ReadData["prediction_date"].map(lambda x: x.year)
    ReadData["Month"]=ReadData["prediction_date"].map(lambda x: x.month)
    ReadData["Day"]=ReadData["prediction_date"].map(lambda x: x.day)
    ReadData["Hour"]=ReadData["prediction_date"].map(lambda x: x.hour)
    del ReadData["prediction_date"]

    if f==0:
        Data = ReadData
    else:
        Data = pd.concat([Data,ReadData])


##
#Data["available_date"]=pd.to_datetime(Data["available_date"])
#Data["prediction_date"]=pd.to_datetime(Data["prediction_date"])
HourlyData = Data[ Data.point==1]
del HourlyData["point"]
Temp2 = Data[Data.point==2]
del Temp2["point"]
HourlyData = pd.merge(HourlyData,Temp2,on=["available_date","Year","Month","Day","Hour"],suffixes=('1','2'))
for t in range(3,19):
    Temp2 = Data[Data.point==t]
    del Temp2["point"]
    HourlyData = pd.merge(HourlyData,Temp2,on=["available_date","Year","Month","Day","Hour"],suffixes=('',str(t)))

##
HourlyData .rename(columns={'wind_speed_100m': 'wind_speed_100m3',
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
IndexList = np.array(range(0,HourlyData.shape[0])).reshape(1,-1)
IndexList = np.reshape(IndexList,(len(FileList),7,24))


##
Transfer = [0] * 7
for h in range(0,7):
    Selection = IndexList[:,h,:].reshape(1,-1)
    Selection = Selection.reshape(-1)
    Transfer[h]=HourlyData.iloc[Selection,:]




###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$




##### Price Information ###############


WeatherFileLoc="/Users/dueheelee/Documents/ComPlatt/Data/Historical_Price/"
RawPrice  = pd.read_csv(WeatherFileLoc + "NewPrice.csv")
#RawPrice["Date"]=pd.to_datetime(RawPrice["Date"])
#RawPrice .rename(columns={'Date': 'prediction_date'},inplace=True)
































#    Template = pd.merge(Outputs_ECM[ColNames],Final7[ColNames],on=SixTag,suffixes=("_ECM","_MY"))








##A=datetime.datetime.strptime(Data.prediction_date[1:2], "%Y-%m-%d %H:%M:%S")


# Final45=pd.concat([SilverCata_AreaState , NoSilverNoCata_AreaState])




# A=datetime.strptime(Temp.available_date[1],"%Y-%m-%d %H:%M:%S")




#FileList = os.listdir(WeatherFileLoc + TargetFileName)





#Original = pd.read_csv(Folder+FileName)			# Read a file



