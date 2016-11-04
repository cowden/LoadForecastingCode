#######################################################################
## Import Packages
#######################################################################
# Basic Packages
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
## Main Function
#######################################################################
def PipeMain():
	global Param
	
	## User Parameter Calling
	Param=ParameterSetting()

	## Casse Specific Preprocessing
	DataPath="/Users/dueheelee/Documents/PyApp/DataMart/PipePricing/"
	TrainX, TrainY, TestsX =PreProcessing(DataPath)

	## Pre Visualization
	PreVisualization(TrainX,TrainY)

	## The First Cross Validation
	FunRMSE, SavePredict = CrossVal(Param,TrainX,TrainY)
	
	## Ensemble Prediction
	TrainOutput,Weight,BestGroup,BestMachine=Ensemble(Param,FunRMSE,SavePredict,TrainY)

	## The First Final Training
	SaveForecast=FinalTraining(Param,TrainX, TrainY, TestsX, Weight, BestGroup, BestMachine)

	## PostProcessing and Printing
	Answer1=PostPrinting(SaveForecast, 'Sub1.csv')

	## Preparing Second Race
	NewTrainX, NewTestsX = SecondWave(TrainOutput,SaveForecast,TrainX,TestsX)

	## The Second Cross Validation
	FunRMSE2, SavePredict2 = CrossVal(Param,NewTrainX,TrainY)

	## Ensemble Prediction
	TrainOutput,Weight2,BestGroup2,BestMachine2=Ensemble(Param,FunRMSE2,SavePredict2,TrainY)

	## The Second Final Training
	SaveForecast2=FinalTraining(Param,NewTrainX,TrainY,NewTestsX,Weight2, BestGroup2, BestMachine2)

	## PostProcessing and Printing
	Answer2=PostPrinting(SaveForecast2, 'Sub2.csv')
























#######################################################################
## Case Dependent Parameter Loading
#######################################################################
def ParameterSetting():
	# Cross Validation Parameters
	Param={'CVN': 3, 'CVOptExternal':"HoldOut"}

	# Machine List
	Param['MachineList']=['XGB','RF']   #  RF GBM XGB

	# Group Number
	Param['NGroup']=[1,6,7]  # ,3,4,6
	
	# Ensem MEthod
	Param['Ensem']="Weight"
	
	# GBM
	Param['GBM_NTry']=2000  # 4000
	Param['GBM_LRate']=0.07
	Param['GBM_CVP']=0.1
	
	# Random Forest
	Param['RF_NTree']=10
	Param['RF_CVP']=0.1
	
	# SVM
	Param['SVM_CVP']=0.1
	Param['C']=[0.1,10,100,300,500,1000,10000,100000]
	Param['Gamma']=0.0
	
	# KNN
	Param['NNeighbors']=10
	
	# Ridge Regression
	Param['Reg_Reg']=[0.0001, 0.001, 0.01,0.1,0.5,1,5,10,100,1000]
	Param['Reg_CVN']=1
	Param['Reg_CVP']=0.1
	
	# Min and Max for preprocessing
	Param['Min']=0.4055
	Param['Max']=6.9088
	
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
	Param['XGB_NR']=10   # 300


	return Param















#######################################################################
## Preprocessing
#######################################################################
def PreProcessing(DataPath):
	DataSaver, FileList = DataLoading(DataPath)
	
	## Bring Files to Data
	TrainOrigin=DataSaver[15]
	TestsOrigin=DataSaver[14]




	###################################
	# Tube
	Bill=DataSaver[0]
	Specs=DataSaver[13]
	Tube=DataSaver[16]
	
	TubeList=pd.merge(Bill , Specs , on='tube_assembly_id')
	TubeList=pd.merge(TubeList , Tube , on='tube_assembly_id')
	TubeList=TubeList.sort(['tube_assembly_id'])
	
	
	Count=TubeList.loc[:,['component_id_1','component_id_2','component_id_3','component_id_4','component_id_5','component_id_6','component_id_7','component_id_8']]
	CountNullSum=8-Count.isnull().sum(axis=1)
	CountNullSum.name='count'

	CpNumber=TubeList.loc[:,['quantity_1','quantity_2','quantity_3','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8']]
	CpNumberSum=CpNumber.sum(axis=1)
	CpNumberSum.name='sum'

	TubeList=pd.concat([TubeList,CountNullSum,CpNumberSum],axis=1)






	###################################
	# Component
	Components=DataSaver[12]
	Components=Components.iloc[:,[0,2]]
	Other=DataSaver[7]
	Other=Other.iloc[:,[0,2]]
	CompList=pd.merge(Components,Other,how='outer',on='component_id')

	for i in [1,2,3,4,5,6,8,9,10,11]:
		CompList=pd.merge(CompList,DataSaver[i],how='outer',on='component_id')

	# type_component
	# CompList=pd.merge(CompList , DataSaver[18],how='outer',on='component_type_id')
	CompList=CompList.sort(['component_id'])
	CompList.index=np.arange(0,len(CompList))





	###################################
	##	## ExtTube
	ExtTubeList=TubeList.copy()
	for i in range(1,5):
		Title='component_id_'+str(i)
		CompIndex=TubeList[Title].values
		CompIndex=CompIndex-1
		Extracted=CompList.loc[CompIndex,:]
		Extracted.index=TubeList.index
		ExtTubeList=pd.concat([ExtTubeList,Extracted],axis=1)






	###################################
	# Null Column Detector
	NullDetector=ExtTubeList.isnull().sum()
	NullDetector=NullDetector.values
	#Index=[NullDetector!=ExtTubeList.shape[0]][0]
	Index=[NullDetector < 3000][0]  # Fix
	ExtTubeList=ExtTubeList.iloc[:,Index]







	###################################
	## Train and Test Data Generation
	TrainXOrigin=TrainOrigin.iloc[:,[0,1,2,3,4,5,6,7,8,9]]
	TrainY=TrainOrigin.cost
	TestsXOrigin=TestsOrigin.iloc[:,[1,2,3,4,5,6,7,8,9,10]]

	# Safe Copy
	TrainXOrigin=TrainXOrigin.copy()
	TrainY=TrainY.copy()
	TrainY=np.log(TrainY+1)
	TestsXOrigin=TestsXOrigin.copy()

	# Inverse QUantityds
	TrainXOrigin['quantity']=1/TrainXOrigin['quantity']
	TrainXOrigin['quantity']=1/TrainXOrigin['quantity']





	###################################
	# Tube Information Extend
	ExtTrainX=pd.merge(TrainXOrigin,ExtTubeList,how='left',on='tube_assembly_id')
	ExtTestsX=pd.merge(TestsXOrigin,ExtTubeList,how='left',on='tube_assembly_id')

	SquareDiameterTrain=pd.DataFrame({'squarediameter': (ExtTrainX['diameter'].values)**2,
									  'diff': (ExtTrainX['diameter'].values-ExtTrainX['wall'].values),
									  'surface':(ExtTrainX['diameter'].values*ExtTrainX['length'].values)})
	SquareDiameterTests=pd.DataFrame({'squarediameter': (ExtTestsX['diameter'].values)**2, 
									  'diff': (ExtTestsX['diameter'].values-ExtTestsX['wall'].values),
									  'surface':(ExtTestsX['diameter'].values*ExtTestsX['length'].values)})

	ExtTrainX=pd.concat([ExtTrainX,SquareDiameterTrain],axis=1)
	ExtTestsX=pd.concat([ExtTestsX,SquareDiameterTests],axis=1)





	######################################################################
	# Reindexing
	#ExtTrainX=ExtTrainX.sort(['tube_assembly_id'])
	ExtTrainX.index=np.arange(0,len(ExtTrainX))
	TrainY.index=np.arange(0,len(TrainY))
	#ExtTestsX=ExtTestsX.sort(['tube_assembly_id'])
	ExtTestsX.index=np.arange(0,len(ExtTestsX))





	######################################################################
	# Printing for just a case
	ExtTubeList.to_csv('Output.csv',index=False,header=True)
	ExtTrainX.to_csv('ExtTrainX.csv',index=False,header=True)
	ExtTestsX.to_csv('ExtTestsX.csv',index=False,header=True)





	######################################################################
	## Feature Engineering
	ExtTrainX=ExtTrainX.drop('tube_assembly_id',1)
	ExtTestsX=ExtTestsX.drop('tube_assembly_id',1)
	ExtTrainX=ExtTrainX.drop('Month',1)
	ExtTestsX=ExtTestsX.drop('Month',1)




	######################################################################
	## Removing Null Space
	TestsNull=ExtTestsX.isnull()
	for i in range(0,TestsNull.shape[1]):
		Temp=TestsNull.iloc[:,i].values
		ExtTestsX.iloc[Temp,i]=-1
	
	return ExtTrainX, TrainY, ExtTestsX



















#######################################################################
## Grouping  [ Case Dependent ]
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
		NG=2

		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		TrainGroup[0]=TrainX[TrainX['Forecast']< np.log(50+1) ].index
		TestsGroup[0]=TestsX[TestsX['Forecast']< np.log(50+1) ].index

		TrainGroup[1]=TrainX[TrainX['Forecast']>=np.log(50+1) ].index
		TestsGroup[1]=TestsX[TestsX['Forecast']>=np.log(50+1) ].index

		print len(TrainGroup[0]), len(TrainGroup[1])



	elif Decision==3:
		NG=2

		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		TrainGroup[0]=TrainX[TrainX['count']<=2 ].index
		TestsGroup[0]=TestsX[TestsX['count']<=2 ].index

		TrainGroup[1]=TrainX[TrainX['count']>=3 ].index
		TestsGroup[1]=TestsX[TestsX['count']>=3 ].index
		
		print len(TrainGroup[0]), len(TrainGroup[1])




	elif Decision==4:  ## Lets do this.
		NG=2

		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		TrainGroup[0]=TrainX[TrainX['bracket_pricing']==1 ].index
		TestsGroup[0]=TestsX[TestsX['bracket_pricing']==1 ].index

		TrainGroup[1]=TrainX[TrainX['bracket_pricing']==0 ].index
		TestsGroup[1]=TestsX[TestsX['bracket_pricing']==0 ].index
		
		print len(TrainGroup[0]), len(TrainGroup[1])




	elif Decision==5:   # Week and Hour
		NG=4
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG
		
		TrainGroup[0]=TrainX[TrainX['Year'] <= 2008 ].index
		TestsGroup[0]=TestsX[TestsX['Year'] <= 2008 ].index
		
		TrainGroup[1]=TrainX[(TrainX['Year'] >= 2009) & (TrainX['Year'] <= 2012)].index
		TestsGroup[1]=TestsX[(TestsX['Year'] >= 2009) & (TestsX['Year'] <= 2012)].index

		TrainGroup[2]=TrainX[(TrainX['Year'] >= 2013) & (TrainX['quantity_1']==1)].index # Bad Performance
		TestsGroup[2]=TestsX[(TestsX['Year'] >= 2013)  & (TestsX['quantity_1']==1)].index # Bad Performance
	
		TrainGroup[3]=TrainX[(TrainX['Year'] >= 2013) & (TrainX['quantity_1']>=2)].index
		TestsGroup[3]=TestsX[(TestsX['Year'] >= 2013)  & (TestsX['quantity_1']>=2)].index





	elif Decision==6:
		NG=2
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG
		
		TrainGroup[0]=TrainX[TrainX['Year'] <= 2012 ].index
		TestsGroup[0]=TestsX[TestsX['Year'] <= 2012 ].index

		TrainGroup[1]=TrainX[(TrainX['Year'] >= 2013)].index # amazing performance
		TestsGroup[1]=TestsX[(TestsX['Year'] >= 2013)].index

		print len(TrainGroup[0]), len(TrainGroup[1])#, len(TrainGroup[2]), len(TrainGroup[3])
	




	elif Decision==7:
		NG=5
		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		TrainGroup[0]=TrainX[TrainX['Year'] <= 2010 ].index
		TestsGroup[0]=TestsX[TestsX['Year'] <= 2010 ].index

		TrainGroup[1]=TrainX[TrainX['Year'] == 2011 ].index
		TestsGroup[1]=TestsX[TestsX['Year'] == 2011 ].index

		TrainGroup[2]=TrainX[TrainX['Year'] == 2012 ].index
		TestsGroup[2]=TestsX[TestsX['Year'] == 2012 ].index

		TrainGroup[3]=TrainX[TrainX['Year'] == 2013].index # amazing performance
		TestsGroup[3]=TestsX[TestsX['Year'] == 2013].index

		TrainGroup[4]=TrainX[(TrainX['Year'] >= 2014)].index # amazing performance
		TestsGroup[4]=TestsX[(TestsX['Year'] >= 2014)].index

		print len(TrainGroup[0]), len(TrainGroup[1]), len(TrainGroup[2]), len(TrainGroup[3]), len(TrainGroup[4])






	elif Decision==8:  ## Lets do this.
		NG=4

		TrainGroup=[0]*NG
		TestsGroup=[0]*NG

		TrainGroup[0]=TrainX[(TrainX['end_x']==1) & (TrainX['end_a']==1) ].index
		TestsGroup[0]=TestsX[(TestsX['end_x']==1) & (TestsX['end_a']==1) ].index

		TrainGroup[1]=TrainX[(TrainX['end_x']==0) & (TrainX['end_a']==1) ].index
		TestsGroup[1]=TestsX[(TestsX['end_x']==0) & (TestsX['end_a']==1) ].index

		TrainGroup[2]=TrainX[(TrainX['end_x']==1) & (TrainX['end_a']==0) ].index
		TestsGroup[2]=TestsX[(TestsX['end_x']==1) & (TestsX['end_a']==0) ].index

		TrainGroup[3]=TrainX[(TrainX['end_x']==0) & (TrainX['end_a']==0) ].index
		TestsGroup[3]=TestsX[(TestsX['end_x']==0) & (TestsX['end_a']==0) ].index

		print len(TrainGroup[0]), len(TrainGroup[1]), len(TrainGroup[2]), len(TrainGroup[3])



	return NG, TrainGroup, TestsGroup



























########################################################################
# Run as a main function
########################################################################
if __name__ == "__main__":
	from Regression import CrossVal, Ensemble, FinalTraining, PreVisualization
	from Regression import SecondWave, PostPrinting, DataLoading
	PipeMain()


#	try:
#		PipeMain()
#	except:
#		type, value, tb = sys.exc_info()
#        traceback.print_exc()
#        pdb.post_mortem(tb)




