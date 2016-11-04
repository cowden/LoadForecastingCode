import sys
import json
from sys import path
import pandas as pd
import numpy as np
import os
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pdb, traceback
import datetime
import time










#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def CrimeMain():
	global Param
	
	## User Case Preprocessing
	DataPath="/Users/dueheelee/Documents/PyApp/DataMart/CrimeClassification/"
	TrainX, TrainY, TestsX, NClass, UniqueTerm=PreProcessing(DataPath)
	
	## Visualization
	PreVisualization(TrainX, TrainY, NClass, UniqueTerm)
	
	## User Parameter Calling
	Param=ParameterSetting(NClass)

	## Cross Validation Ticket Generation
	FunRMSE, SavePredict=CrossVal(Param,TrainX,TrainY)

	## Select the Ensemble Function and Weights
	TrainOutput,Weight,BestGroup,BestMachine=Ensemble(Param,FunRMSE,SavePredict,TrainY)

	## The First Final Training
	SaveForecast=FinalTraining(Param,TrainX, TrainY, TestsX, Weight, BestGroup, BestMachine)

	## PostProcessing and Printing
	PostPrinting(SaveForecast, UniqueTerm, 'Sub1.csv')

	## Preparing Second Race
	NewTrainX, NewTestsX = SecondWave(TrainOutput,SaveForecast,TrainX,TestsX)
	
	## The Second Cross Validation
	FunRMSE2, SavePredict2 = CrossVal(Param,NewTrainX,TrainY)

	## Ensemble Prediction
	TrainOutput2,Weight2,BestGroup2,BestMachine2=Ensemble(Param,FunRMSE2,SavePredict2,TrainY)

	## The Second Final Training
	SaveForecast2=FinalTraining(Param,NewTrainX,TrainY,NewTestsX,Weight2, BestGroup2, BestMachine2)

	## PostProcessing and Printing
	PostPrinting(SaveForecast2,UniqueTerm, 'Sub2.csv')




















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


















#######################################################################
## Preprocessing
#######################################################################
def PreProcessing(DataPath):
	# Load Data
	DataSaver, FileList = DataLoading(DataPath)
	TrainOrigin=DataSaver[1]        # Train
	TestOrigin=DataSaver[0]         # Test


	## Select Columns
	TrainXOrigin=TrainOrigin.iloc[:,[0,1,2,3,4,5,8,9,12,13]]
	TrainYOrigin=TrainOrigin.Category
	TestsXOrigin=TestOrigin.iloc[:,[1,2,3,4,5,6,7,8,10,11]]

	TrainXOrigin=TrainXOrigin.copy()
	TrainYOrigin=TrainYOrigin.copy()
	TestsXOrigin=TestsXOrigin.copy()
	
	## String Converter
	# TrainX and TestsX
	Total=pd.concat([TrainXOrigin,TestsXOrigin],axis=0)
	ColumnList=["PdDistrict"]  # ,"Address"
	for i in range(0,len(ColumnList)):
		ColumnName=ColumnList[i]
		UniqueTerm=Total[ColumnName].unique()
		print UniqueTerm
		for j in range(0,len(UniqueTerm)):
			Term=UniqueTerm[j]
			TrainXOrigin.loc[TrainXOrigin[ColumnName]==Term,ColumnName]=j
			TestsXOrigin.loc[TestsXOrigin[ColumnName]==Term,ColumnName]=j

	# Weekly Unique Term
	Week=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
	for j in range(0,len(Week)):
		Term=Week[j]
		TrainXOrigin.loc[TrainXOrigin["DayOfWeek"]==Term,"DayOfWeek"]=j
		TestsXOrigin.loc[TestsXOrigin["DayOfWeek"]==Term,"DayOfWeek"]=j


	# TrainY
	UniqueTerm=TrainYOrigin.unique()
	NClass=len(UniqueTerm)
	for j in range(0,len(UniqueTerm)):
		Term=UniqueTerm[j]
		TrainYOrigin[TrainYOrigin==Term]=j

	# Reduce the system size to save some time
#	List=np.random.permutation(TrainXOrigin.shape[0])
#	TrainX=TrainXOrigin.loc[List[:10000],:]
#	TrainY=TrainYOrigin.loc[List[:10000]]
#	TestsX=TestsXOrigin.loc[List[:1000],:]

	TrainX=TrainXOrigin.copy()
	TrainY=TrainYOrigin.copy()
	TestsX=TestsXOrigin.copy()

	TotalMinTrainX=TrainX["Hour"].values*60 + TrainX["Minute"].values
	TotalMinTrainX=pd.Series(TotalMinTrainX,name='TotalMin')
	TrainX=pd.concat([TrainX,TotalMinTrainX],axis=1)

	TotalMinTestsX=TestsX["Hour"].values*60 + TestsX["Minute"].values
	TotalMinTestsX=pd.Series(TotalMinTestsX,name='TotalMin')
	TestsX=pd.concat([TestsX,TotalMinTestsX],axis=1)
	print "End of Preprocessing"

	return TrainX, TrainY, TestsX, NClass, UniqueTerm















#######################################################################
## Grouping
#######################################################################
def GroupGen(Decision,TrainX,TrainY,TestsX):
    # The List containing the group index

	# Group 
	if Decision==1:
		NG=1
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG	
		TrainGroup[0]=np.arange(0,TrainX.shape[0])
		TestsGroup[0]=np.arange(0,TestsX.shape[0])







	elif Decision==2:
		NG=7  # Weekly
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		for week in range(0,7):
			TrainGroup[week]=TrainX[TrainX['DayOfWeek']==week].index
			TestsGroup[week]=TestsX[TestsX['DayOfWeek']==week].index







	elif Decision==3:
		NG=12 # Monthly

		TrainGroup=[0]*NG
		TestsGroup=[0]*NG
		for month in range(1,13):
			TrainGroup[month-1]=TrainX[TrainX['Month']==month].index
			TestsGroup[month-1]=TestsX[TestsX['Month']==month].index





	elif Decision==4:
		NG=12*7 # Month and week

		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		GroupTag=0
		for month in range(1,13):
			SubTrainGroup=TrainX[TrainX['Month']==month]
			SubTestsGroup=TestsX[TestsX['Month']==month]

			for week in range(0,7):
				TrainGroup[GroupTag]=SubTrainGroup[SubTrainGroup['DayOfWeek']==week].index
				TestsGroup[GroupTag]=SubTestsGroup[SubTestsGroup['DayOfWeek']==week].index
				GroupTag=GroupTag+1







	elif Decision==5:   # Week and Hour
		NG=7*8
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG
		GroupTag=0
		for week in range(0,7):
			for hour in range(0,8):
				TrainGroup[GroupTag]=TrainX[(TrainX['DayOfWeek']==week) &
											((TrainX['Hour']==hour*3) |
											(TrainX['Hour']==hour*3+1)|
											(TrainX['Hour']==hour*3+2))].index
				TestsGroup[GroupTag]=TestsX[(TestsX['DayOfWeek']==week) &
											((TestsX['Hour']==hour*3) |
											(TestsX['Hour']==hour*3+1)|
											(TestsX['Hour']==hour*3+2))].index
				GroupTag=GroupTag+1






	elif Decision==6:
		NG=12*7*2
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG
		GroupTag=0
		for month in range(1,13):
			for week in range(0,7):
				for hour in range(0,2):
					if hour==0:
						TrainGroup[GroupTag]=TrainX[(TrainX['Month']==month) &
													(TrainX['DayOfWeek']==week) &
													((TrainX['Hour']>=18)|(TrainX['Hour']<6))].index
						TestsGroup[GroupTag]=TestsX[(TestsX['Month']==month) &
													(TestsX['DayOfWeek']==week) &
													((TestsX['Hour']>=18)|(TestsX['Hour']<6))].index

					elif hour==1:
						TrainGroup[GroupTag]=TrainX[(TrainX['Month']==month) &
													(TrainX['DayOfWeek']==week) &
													(TrainX['Hour']>=6) &
													(TrainX['Hour']<18) 		].index
						TestsGroup[GroupTag]=TestsX[(TestsX['Month']==month) &
													(TestsX['DayOfWeek']==week) &
													(TestsX['Hour']>=6) &
													(TestsX['Hour']<18) 		].index
					GroupTag=GroupTag+1







	elif Decision==7:
		NG=10  # PDistrict
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		for week in range(0,10):
			TrainGroup[week]=TrainX[TrainX['PdDistrict']==(week)].index
			TestsGroup[week]=TestsX[TestsX['PdDistrict']==(week)].index








	elif Decision==8:
		NG=10*7*2        # District * Week * Hour
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG
		GroupTag=0
		for pd in range(0,10):
			for week in range(0,7):
				for hour in range(0,2):
					if hour==0:
						TrainGroup[GroupTag]=TrainX[(TrainX['PdDistrict']==pd) &
													(TrainX['DayOfWeek']==week) &
													((TrainX['Hour']>=18)|(TrainX['Hour']<6))].index
						TestsGroup[GroupTag]=TestsX[(TestsX['PdDistrict']==pd) &
													(TestsX['DayOfWeek']==week) &
													((TestsX['Hour']>=18)|(TestsX['Hour']<6))].index

					elif hour==1:
						TrainGroup[GroupTag]=TrainX[(TrainX['PdDistrict']==pd) &
													(TrainX['DayOfWeek']==week) &
													(TrainX['Hour']>=6) &
													(TrainX['Hour']<18) 		].index
						TestsGroup[GroupTag]=TestsX[(TestsX['PdDistrict']==pd) &
													(TestsX['DayOfWeek']==week) &
													(TestsX['Hour']>=6) &
													(TestsX['Hour']<18) 		].index
					GroupTag=GroupTag+1







	elif Decision==9:
		NG=10*7*4        # District * Week * Hour
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG
		GroupTag=0
		for pd in range(0,10):
			for week in range(0,7):
				for hour in range(0,4):
					TrainGroup[GroupTag]=TrainX[(TrainX['PdDistrict']==pd) &
												(TrainX['DayOfWeek']==week) &
												((TrainX['Hour']>=hour*6)|(TrainX['Hour']<(hour+1)*6))].index
					TestsGroup[GroupTag]=TestsX[(TestsX['PdDistrict']==pd) &
												(TestsX['DayOfWeek']==week) &
												((TestsX['Hour']>=hour*6)|(TestsX['Hour']<(hour+1)*6))].index

					GroupTag=GroupTag+1


	return NG, TrainGroup, TestsGroup














#######################################################################
## Run Mail
#######################################################################
# Run as a main function
if __name__ == "__main__":
	from Classification import CrossVal, Ensemble, FinalTraining, PreVisualization
	from Classification import SecondWave, PostPrinting, DataLoading
	CrimeMain()



# Debugging Mode
#	try:
#		CrimeMain()
#	except:
#		type, value, tb = sys.exc_info()
#        traceback.print_exc()
#        pdb.post_mortem(tb)

















