

def ParameterSetting():

	Param={'Ridge_Alpha':[0.001,0.01,0.1,1,10,100]}
	Param['RF_NumTree']=2000

	Param['CVOpt']='HoldOut'
	
	
	
	# GBM
	Param['GBM_NTry']=2000
	Param['GBM_LRate']=0.01
	
	
	
	# Random Forest
	Param['RF_NTree']=300
	


	return Param







#
#import sys
#import pandas as pd
#import numpy as np
#import os
#import pdb
#import matplotlib.pyplot as plt
#
########################################################################
### User Define Function Calling
########################################################################
#from UD import GroupGen
#from UD import Preprocessing
#from UD import Machinery
#
#from FunParam import ParameterSetting
########################################################################
#
#
########################################################################
### Clear Variables
########################################################################
#clear= lambda:  os.system('cls')
#clear()
#################################################
#
########################################################################
### Set Important Parameters 
########################################################################
#CVN=2
#CVP=0.2
#MachineList=['Ridge','RF']
#GroupDecision=1
#EnsemOption='Best'
#
## Bring internal parameters for forecasting machines
#Param=ParameterSetting()
########################################################################
#
#
########################################################################
### Weekly Setting
########################################################################
#Week=5
#if Week < 10:
#    WeekFileName="0"+str(Week)
########################################################################
#
#
########################################################################
### Input Data Receiver
########################################################################
## Path Information
#DataPath="/Users/dueheelee/Documents/M/Solar/Data/"
#
## File Information
#PredictorFile="predictors"+WeekFileName+".csv"
#TrainFile="train"+WeekFileName+".csv"
#SolutionFile="Sol"+WeekFileName+".csv"
#
## Read CSV Files
#PredictorOrigin=pd.read_csv(DataPath+PredictorFile)
#TrainOrigin=pd.read_csv(DataPath+TrainFile)
#Sol=pd.read_csv(DataPath+SolutionFile,header=None)
########################################################################
#
#
#
#
########################################################################
### Zone Selection
########################################################################
## First zone Detection
#Zone=1
#Pre1=PredictorOrigin[PredictorOrigin.ZONEID==1]
##Pre1.columns=Pre1.columns+'_2'
#
## Zone 2 Handling
#Pre2=PredictorOrigin[PredictorOrigin.ZONEID==2]
#Pre2=Pre2.iloc[:,5:]
#Pre2.columns=Pre2.columns+'_2'
#Pre2.index=Pre1.index
#
## Zone 3 Handling
#Pre3=PredictorOrigin[PredictorOrigin.ZONEID==3]
#Pre3=Pre3.iloc[:,5:]
#Pre3.columns=Pre3.columns+'_3'
#Pre3.index=Pre1.index
#
#
#Pre=pd.concat([Pre1,Pre2,Pre3],axis=1)
#Trn=TrainOrigin=TrainOrigin[TrainOrigin.ZONEID==Zone]
#
### Feature Engineering
#TrainX=Pre.iloc[0:Trn.shape[0],1:]
#TestsX=Pre.iloc[Trn.shape[0]:,1:]
#TrainY=Trn.POWER
#
## Reset the index
#TrainX.index=np.arange(0,TrainX.shape[0])
#TestsX.index=np.arange(0,TestsX.shape[0])
#TrainY.index=np.arange(0,TrainY.shape[0])
#
## Sol Selection
#Sol=Sol[(Zone-1)*744:Zone*744]
########################################################################
#
#
#
########################################################################
### Daylight Index Detector
########################################################################
## Target Month
#TargetMonth=Trn.Month.iloc[-1]
#
## Index of days on testx
#TargetMonthList=TrainX[TrainX.Month == TargetMonth].iloc[:-1,:]
#TargetDay=TrainY[TargetMonthList.index[1]:TargetMonthList.index[-1]+2]
#TestDayIndex=np.where(TargetDay > 0 )[0]
#
#TrainXDay=TrainX[TrainY.values > 0]
#TrainYDay=TrainY[TrainY.values > 0]
#TestsXDay=TestsX.iloc[TestDayIndex,:]
########################################################################
#
#
#
#
########################################################################
### Outlier Detection
########################################################################
#
#
#
########################################################################
#
#
#
#
########################################################################
### Cross Validation Ticket Generation
########################################################################
#LTrain=TrainXDay.shape[0]					# Length of the total training data
#TotalIndex=np.arange(0,LTrain)				# Index of total set
#LTestCV=np.round(CVP*LTrain)				# Length of Test set in the CV
#
#FunRMSE=np.zeros( (CVN,len(MachineList)) )
#SavePredict=np.zeros( (CVN,len(MachineList), LTestCV) )
#for cvindex in range(0,CVN):
#	TestSelTemp=np.random.permutation(LTrain)
#	TestsSel=TestSelTemp[:LTestCV]
#	TestsSel.sort()
#	TrainSel=np.setdiff1d(TotalIndex,TestsSel)
#
#	# Select the training data and testing data for cross validation
#	SubTrainX=TrainXDay.iloc[TrainSel,:]
#	SubTrainY=TrainYDay.iloc[TrainSel]
#	SubTestsX=TrainXDay.iloc[TestsSel,:]
#	SubTestsY=TrainYDay.iloc[TestsSel]
#
#	for machindex in range(0,len(MachineList)):
#		Machine=MachineList[machindex]
#		Prediction=np.zeros(len(SubTestsY))
#
#		# Group Data
#		# NG: Number of Groups
#		# TrainGroup: List of indexes of each group for training data
#		# TestsGroup: List of indexes of each group for testing data
#		NG,TrainGroup,TestsGroup= GroupGen(GroupDecision,SubTrainX, SubTrainY, SubTestsX)
#		
#		for grindex in range(0,NG):
#
#			# Group Assigning
#			TrnX=SubTrainX.iloc[TrainGroup[grindex],:]
#			TrnY=SubTrainY.iloc[TrainGroup[grindex]]
#			TstX=SubTestsX.iloc[TestsGroup[grindex],:]
#			TstY=SubTestsY.iloc[TestsGroup[grindex]]
#
#			# Preprocessing
#			STrnX,STstX=Preprocessing(TrnX, TstX)
#
#			# Forecast
#			Forecast=Machinery(Machine,Param,STrnX,TrnY,STstX)
#			
#			# Assigning
#			Prediction[TestsGroup[grindex]]=Forecast
#
#
#		FunRMSE[cvindex,machindex]=np.mean((Prediction-SubTestsY.values)**2)
#		SavePredict[cvindex,machindex,:]=Prediction
#	
#		# Preprocessing
#		PostPrediction=np.maximum(Prediction,0)
#		PostPrediction=np.minimum(PostPrediction,1)
#	
#		# Printing
##		X=np.arange(0,PostPrediction.shape[0])
##		plt.plot(X,SubTestsY.values, color="blue",linewidth=1.0,linestyle="-")
##		plt.plot(X,PostPrediction, color="red",linewidth=1.0,linestyle="-")
##		#plt.xlim(-4,4)
##		#plt.xticks(np.linspace(-4,4,9,endpoint=True))
##		plt.title('Test: '+Machine)
##		# plt.show()
#
#		print str(cvindex+1) + "." + Machine + " : " + str(FunRMSE[cvindex,machindex])
#
#
#
#
########################################################################
########################################################################
### Select the Ensemble Function and Weights
########################################################################
#if EnsemOption=='Best':
#	MeanFunRMSE=np.mean(FunRMSE,axis=0)					# Average Performance of each Functions
#	BestMachineIndex=np.argmin(MeanFunRMSE)
#	Temp=BestMachineIndex
#	BestMachineIndex=[]
#	BestMachineIndex.append(Temp)
#	Weight=1
#	
#
#
## Just simply averaging every forecasting machine
#elif EnsemOption=='SimpleAveraging':
#	Temp=1
#	Weight=[1.0/len(BestMachineIndex)]*len(BestMachineIndex)
#
## Find the weights for the weighted averaging
#elif EnsemOption=='WeightAveraging':
#	Temp=1
#
## Find the best combination of multiple forecasting machines by using the simple averaging
#elif EnsemOption=='BestCombSimple':
#	Temp=1
#
## Find the best combination of multiple forecasting machines by using the weighted averaging
#elif EnsemOption=='BestCombWeight':
#	Temp=1
#
#
########################################################################
#
#
#
#######################################################################################
#######################################################################################
### Final Training: All training data is used to train the model
#SavePredict=np.zeros( (len(BestMachineIndex), len(TestsXDay)) )
#for machindex in range(0,len(BestMachineIndex)):
#
#	Machine=MachineList[BestMachineIndex[machindex]]
#	Prediction=np.zeros(len(TestsXDay))
#
#	# Group Data
#	# NG: Number of Groups
#	# TrainGroup: List of indexes of each group for training data
#	# TestsGroup: List of indexes of each group for testing data
#	NG,TrainGroup,TestsGroup= GroupGen(GroupDecision,TrainXDay, TrainYDay, TestsXDay)
#	
#	for grindex in range(0,NG):
#		# Group Assigning
#		TrnX=TrainXDay.iloc[TrainGroup[grindex],:]
#		TrnY=TrainYDay.iloc[TrainGroup[grindex]]
#		TstX=TestsXDay.iloc[TestsGroup[grindex],:]
#
#		# Preprocessing
#		STrnX,STstX=Preprocessing(TrnX, TstX)
#
#		# Forecast
#		Forecast=Machinery(Machine,Param,STrnX,TrnY,STstX)
#		
#		# Assigning
#		Prediction[TestsGroup[grindex]]=Forecast
#
#	SavePredict[machindex,:]=Prediction
##########################################################################################
#
#
#
##########################################################################################
##########################################################################################
### Determine the Ensemble Result
##########################################################################################
## Weighted Ensemble
#WeightedOutput=np.dot(SavePredict,Weight)
##########################################################################################
#
#
##########################################################################################
### Put data on a day light
##########################################################################################
#Final=np.zeros(len(TestsX))
#Final[TestDayIndex]=WeightedOutput
##########################################################################################
#
#
#
##########################################################################################
## Postprocessing
##########################################################################################
#Final=np.maximum(Final,0)
#Final=np.minimum(Final,1)
##########################################################################################
#
#
#
##########################################################################################
##########################################################################################
### Visualization
##########################################################################################
#RMSE=np.mean((Final-Sol.values)**2)
#print RMSE
#
#X=np.arange(0,Final.shape[0])
#plt.plot(X,Sol.values, color="blue",linewidth=1.0,linestyle="-")
#plt.plot(X,Final, color="red",linewidth=1.0,linestyle="-")
##plt.xlim(-4,4)
##plt.xticks(np.linspace(-4,4,9,endpoint=True))
#plt.title('Final : ' + str(RMSE))
#plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
