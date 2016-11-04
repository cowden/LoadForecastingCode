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
from sys import getsizeof as gs









#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def CrimeMain():
	global Param
	
	## User Case Preprocessing
	DataPath="/Users/dueheelee/Documents/PyApp/DataMart/Springleaf/"
	TrainX, TrainY, TestsX, NClass, UniqueTerm=PreProcessing(DataPath)
	
	
	bp()
	
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


	for s in range(1,3):   # 1: Test, 2: Train
		Select=DataSaver[s]
	
		# Time Information Remover
		del Select['VAR_0073'], Select['VAR_0075'], Select['VAR_0156'], Select['VAR_0159']
		del Select['VAR_0166'], Select['VAR_0169'], Select['VAR_0176'], Select['VAR_0177']
		del Select['VAR_0178'], Select['VAR_0179'], Select['VAR_0204'], Select['VAR_0217']


		# String Remover
		del Select['VAR_0200']
		

		# Single Value Remover
		Col=Select.columns
		for i in range(0,len(Col)):
			print i
			U=Select[Col[i]].unique()
			NullTester=pd.isnull(Select[Col[i]])
			NNull=sum(NullTester)

			if len(U)==1:
				print Col[i], U
				del Select[Col[i]]
				continue
			elif (len(U)==2) & ( (pd.isnull(U[0])==True) | (pd.isnull(U[1])==True) ):
				print Col[i], U
				del Select[Col[i]]
				continue
			elif NNull > 10000:
				del Select[Col[i]]
				continue
			

#			Select.loc[Select[Col[i]]==999999999,Col[i]]=-9
#			Select.loc[Select[Col[i]]==999999998,Col[i]]=-8
#			Select.loc[Select[Col[i]]==999999997,Col[i]]=-7
#			Select.loc[Select[Col[i]]==999999996,Col[i]]=-6
#			Select.loc[Select[Col[i]]==999999995,Col[i]]=-5
#			Select.loc[Select[Col[i]]==999999994,Col[i]]=-4
#			Select.loc[Select[Col[i]]==99999,Col[i]]=-3
#			Select.loc[Select[Col[i]]==9999,Col[i]]=-19
#			Select.loc[Select[Col[i]]==9998,Col[i]]=-18
#			Select.loc[Select[Col[i]]==9997,Col[i]]=-17
#			Select.loc[Select[Col[i]]==9996,Col[i]]=-16
#			Select.loc[Select[Col[i]]==9995,Col[i]]=-15
#			Select.loc[Select[Col[i]]==9994,Col[i]]=-14
#			Select.loc[Select[Col[i]]==998,Col[i]]=-13

			Select.loc[Select[Col[i]]==False,Col[i]]=0
			Select.loc[Select[Col[i]]==True,Col[i]]=1

			del U, NullTester, NNull

		DataSaver[s]=Select

	TrainOrigin=DataSaver[2]        # Train
	TestsOrigin=DataSaver[1]         # Test
	
	# Cleaning
	del DataSaver, FileList, Select

	# Convert the str to number
	Total=pd.concat([TrainOrigin,TestsOrigin],axis=0)
	ColumnList=["VAR_0001","VAR_0005","VAR_0237","VAR_0274","VAR_0283",
				"VAR_0305","VAR_0352","VAR_0353","VAR_1934"]
	for i in range(0,len(ColumnList)):
		ColumnName=ColumnList[i]
		UniqueTerm=Total[ColumnName].unique()

		# nan Remover
		NullDetect=pd.isnull(UniqueTerm)
		UniqueTerm=UniqueTerm[~NullDetect]
		print UniqueTerm
		
		for j in range(0,len(UniqueTerm)):
			Term=UniqueTerm[j]
			TrainOrigin.loc[TrainOrigin[ColumnName]==Term,ColumnName]=j
			TestsOrigin.loc[TestsOrigin[ColumnName]==Term,ColumnName]=j

	del Total

	## Fill NA
	TrainOrigin=TrainOrigin.fillna(-1)
	TestOrigin=TestsOrigin.fillna(-1)
	
	## Final Data Preparation
	TrainY=TrainOrigin["target"]
	del TrainOrigin["target"]
	TrainX=TrainOrigin
	TestsX=TestOrigin
	UniqueTerm=TrainY.unique()
	NClass=len(UniqueTerm)

#	TrainOrigin.iloc[:50000,:].to_csv('NewTrain1.csv',index=False,header=True)
#	TrainOrigin.iloc[50000:100000,:].to_csv('NewTrain2.csv',index=False,header=True)
#	TrainOrigin.iloc[100000:,:].to_csv('NewTrain3.csv',index=False,header=True)



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
#	CrimeMain()



# Debugging Mode
	try:
		CrimeMain()
	except:
		type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

















#	# 8 ~ 12
#	Index0008=TrainOrigin[TrainOrigin['VAR_0008']==False].index
#	TrainOrigin.loc[Index0008,'VAR_0008']=0
#	
#	Index0009=TrainOrigin[TrainOrigin['VAR_0009']==False].index
#	TrainOrigin.loc[Index0009,'VAR_0009']=0
#	
#	Index0010=TrainOrigin[TrainOrigin['VAR_0010']==False].index
#	TrainOrigin.loc[Index0010,'VAR_0010']=0
#
#	Index0011=TrainOrigin[TrainOrigin['VAR_0011']==False].index
#	TrainOrigin.loc[Index0011,'VAR_0011']=0
#
#	Index0012=TrainOrigin[TrainOrigin['VAR_0012']==False].index
#	TrainOrigin.loc[Index0012,'VAR_0012']=0
#
#	Index0043=TrainOrigin[TrainOrigin['VAR_0043']==False].index
#	TrainOrigin.loc[Index0043,'VAR_0043']=0
#
#	Index0196=TrainOrigin[TrainOrigin['VAR_0196']==False].index
#	TrainOrigin.loc[Index0196,'VAR_0196']=0
#
#	Index0226=TrainOrigin[TrainOrigin['VAR_0226']==False].index
#	TrainOrigin.loc[Index0226,'VAR_0226']=0
#
#	Index0226=TrainOrigin[TrainOrigin['VAR_0226']==True].index
#	TrainOrigin.loc[Index0226,'VAR_0226']=1
#
#	Index0229=TrainOrigin[TrainOrigin['VAR_0229']==False].index
#	TrainOrigin.loc[Index0229,'VAR_0229']=0
#
#	Index0230=TrainOrigin[TrainOrigin['VAR_0230']==False].index
#	TrainOrigin.loc[Index0230,'VAR_0230']=0
#
#	Index0230=TrainOrigin[TrainOrigin['VAR_0230']==True].index
#	TrainOrigin.loc[Index0230,'VAR_0230']=1
#
#	Index0232=TrainOrigin[TrainOrigin['VAR_0232']==False].index
#	TrainOrigin.loc[Index0232,'VAR_0232']=0
#
#	Index0232=TrainOrigin[TrainOrigin['VAR_0232']==True].index
#	TrainOrigin.loc[Index0232,'VAR_0232']=1
#
#	Index0236=TrainOrigin[TrainOrigin['VAR_0236']==False].index
#	TrainOrigin.loc[Index0236,'VAR_0236']=0
#
#	Index0236=TrainOrigin[TrainOrigin['VAR_0236']==True].index
#	TrainOrigin.loc[Index0236,'VAR_0236']=1
#
#	Index0239=TrainOrigin[TrainOrigin['VAR_0239']==False].index
#	TrainOrigin.loc[Index0239,'VAR_0239']=0

#
#	del TrainOrigin['VAR_0008']
#	del TrainOrigin['VAR_0009']
#	del TrainOrigin['VAR_0010']
#	del TrainOrigin['VAR_0011']
#	del TrainOrigin['VAR_0012']
#	
#	del TrainOrigin['VAR_0018']
#	del TrainOrigin['VAR_0019']
#	del TrainOrigin['VAR_0020']
#	del TrainOrigin['VAR_0021']
#	del TrainOrigin['VAR_0022']
#	del TrainOrigin['VAR_0023']
#	del TrainOrigin['VAR_0024']
#	del TrainOrigin['VAR_0025']
#	del TrainOrigin['VAR_0026']
#	del TrainOrigin['VAR_0027']
#	del TrainOrigin['VAR_0028']
#	del TrainOrigin['VAR_0029']
#	del TrainOrigin['VAR_0030']
#	del TrainOrigin['VAR_0031']
#	del TrainOrigin['VAR_0032']
#	
#	del TrainOrigin['VAR_0038']
#	del TrainOrigin['VAR_0039']
#	del TrainOrigin['VAR_0040']
#	del TrainOrigin['VAR_0041']
#	del TrainOrigin['VAR_0042']
#	del TrainOrigin['VAR_0043']
#	del TrainOrigin['VAR_0044']
#	
#	del TrainOrigin['VAR_0196']
#	
#	del TrainOrigin['VAR_0202']
#	del TrainOrigin['VAR_0203']
#
#
#	del TrainOrigin['VAR_0215']
#	
#
#	del TrainOrigin['VAR_0222']
#	del TrainOrigin['VAR_0223']
#	
#	del TrainOrigin['VAR_0229']
#	del TrainOrigin['VAR_0239']
#
#
#
#
#
#
##	# 8 ~ 12
##	Index0008=TrainOrigin[TrainOrigin['VAR_0008']==False].index
##	TrainOrigin.loc[Index0008,'VAR_0008']=0
#
#	Col=TrainOrigin.columns
#	for i in range(0,TrainOrigin.shape[1]):
#		Temp=TrainOrigin.iloc[:,i]
#		Index=Temp[Temp==False].index
#		TrainOrigin.iloc[Index,i]=0
#
#		Index=Temp[Temp==True].index
#		TrainOrigin.iloc[Index,i]=1
#	
#	
#		Index=Temp[Temp==-9999].index
#		TrainOrigin.iloc[Index,i]=-1
#
#
#		Index=Temp[Temp==999999997].index
#		TrainOrigin.iloc[Index,i]=999999997-1e9
#
#
#		Index=Temp[Temp==999999996].index
#		TrainOrigin.iloc[Index,i]=999999996-1e9
#
#		Index=Temp[Temp==9998].index
#		TrainOrigin.iloc[Index,i]=9998-1e4
#
#
#		Index=Temp[Temp==9997].index
#		TrainOrigin.iloc[Index,i]=9997-1e4
#
#
#		Index=Temp[Temp==9996].index
#		TrainOrigin.iloc[Index,i]=9996-1e4

	
	
#		Unique=Temp.unique()
#		if len(Unique)==2 & Unique[1]==-1
#			del TrainOrigin[Col[i]]
#	
#		if len(Unique)==2 &  Unique[0]==-1
#			del TrainOrigin[Col[i]]



	
	
#	for i in range(0,14):
#		A=TrainOrigin.loc[10000*i:10000*(i+1),:]
#		FileName="A"+str(i)

	
	

