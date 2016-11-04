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

# from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# Figure
import matplotlib.pyplot as plt

## Case Dependent Packages
from Pipe import ParameterSetting , GroupGen




#######################################################################
## Data Loading
#######################################################################
def DataLoading(DataPath):
	FileList=os.listdir(DataPath)
	
	FileList=FileList[1:]
	
	DataSaver=[0] * len(FileList)
	for file in range( 0,len(FileList) ):
		FileName=FileList[file]
		DataSaver[file]=pd.read_csv(DataPath+FileName)

	return DataSaver , FileList









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
def CrossVal(ParamOriginal,TrainX,TrainY):
	global Param
	Param=ParamOriginal

	CVN=Param['CVN']
	MachineList=Param['MachineList']
	GroupDecision=Param['NGroup']

	## Cross Validation
	kf = KFold( len(TrainY) ,shuffle=True, n_folds=CVN )

	## Play the Game
	SavePredict=np.zeros( ( len(MachineList) , len(GroupDecision) , len(TrainY) ) )
	FunRMSE=np.zeros(( len(MachineList) , len(GroupDecision) ))
	for machindex in range(0,len(MachineList)):
		Machine=MachineList[machindex]
		
		for gt in range(0,len(GroupDecision)):
			GD=GroupDecision[gt]

			c=0
			for train_index, test_index in kf:
				c=c+1
				SubTrainX=TrainX.iloc[train_index,:]
				SubTrainY=TrainY.iloc[train_index]
				SubTestsX=TrainX.iloc[test_index,:]
				SubTestsY=TrainY.iloc[test_index]

				# Re-Indexing
				SubTrainX.index=np.arange(0,len(SubTrainX))
				SubTrainY.index=np.arange(0,len(SubTrainY))
				SubTestsX.index=np.arange(0,len(SubTestsX))
				SubTestsY.index=np.arange(0,len(SubTestsY))
				
				NG,TrainGroup,TestsGroup= GroupGen(GD,SubTrainX, SubTrainY, SubTestsX)


				GroupResult=np.zeros( len(SubTestsX))
				for grindex in range(0,NG):
					# Group Assigning
					TrnX=SubTrainX.iloc[TrainGroup[grindex],:]
					TrnY=SubTrainY.iloc[TrainGroup[grindex]]
					TstX=SubTestsX.iloc[TestsGroup[grindex],:]
					TstY=SubTestsY.iloc[TestsGroup[grindex]]

					# Outlier Detection
					TrnX,TrnY=OutlierDetector(TrnX,TrnY)
					
					# Null value Detection
					TrnX=TrnX.copy()
					TstX=TstX.copy()
					TrainNull=TrnX.isnull()
					TestsNull=TstX.isnull()
					for i in range(0,TrainNull.shape[1]):
						Temp=TrainNull.iloc[:,i].values
						TrnX.iloc[Temp,i]=-1
		
						Temp=TestsNull.iloc[:,i].values
						TstX.iloc[Temp,i]=-1

					# Preprocessing
					TrnX,TstX=Standardization(TrnX, TstX)

					# Forecast
					Forecast=Machinery(Machine,TrnX,TrnY,TstX)
					GroupResult[TestsGroup[grindex]]=Forecast
					NormForecast=PostProcessing(Forecast)
					TempPerformance=PerMeasure(TstY,NormForecast)
					print Machine, "Main Group:", GD, "Sub Group:", grindex, "CV:", c, TempPerformance


				SavePredict[machindex, gt, test_index]=GroupResult
				NormForecast=PostProcessing(GroupResult)
				TempPerformance=PerMeasure(SubTestsY,NormForecast)
				print Machine, "Main Group:", GD, "CV:", c, TempPerformance


			SingleOutput=SavePredict[machindex, gt, :]
			NormForecast=PostProcessing(SingleOutput)
			TempPerformance=PerMeasure(TrainY,NormForecast)
			FunRMSE[machindex,gt]=TempPerformance
			print Machine, "Main Group:", GD, TempPerformance


		SingleOutput=SavePredict[machindex, :, :]
		TempResult=np.mean(SingleOutput,axis=0)
		NormForecast=PostProcessing(TempResult)
		TempPerformance=PerMeasure(TrainY,NormForecast)
		print Machine, TempPerformance

	TempResult=np.mean(SavePredict,axis=0)
	TempResult=np.mean(TempResult,axis=0)
	NormForecast=PostProcessing(TempResult)
	Forecast=pd.DataFrame(NormForecast,columns=['Forecast'])
	Error=pd.DataFrame((TrainY.values-NormForecast)**2,columns=['Error'])
	Sample=pd.concat([TrainX,TrainY,Forecast,Error],axis=1)

	# Print Out
	PostVisualization(Sample)
	print FunRMSE, PerMeasure(TrainY,NormForecast)

	return FunRMSE, SavePredict




######################################################################
## Pre Visualization  [ General Case ]
#######################################################################
def PreVisualization(TrainX,TrainY):
	# Feature Importance Extraction
	RF=RandomForestRegressor(n_estimators=1000,verbose=0,min_samples_split=2,
		min_samples_leaf=1,
		bootstrap=True,
		n_jobs=-1)
		
	TrainNull=TrainX.isnull()
	for i in range(0,TrainNull.shape[1]):
		Temp=TrainNull.iloc[:,i].values
		TrainX.iloc[Temp,i]=-1

	RF.fit(TrainX,TrainY)
	FI=RF.feature_importances_
	Data=pd.Series(FI, index=TrainX.columns)
	Data.to_json('PipeFeatureImportance.json')

	return FI




#######################################################################
## Post Visualization  [ General Case ]
#######################################################################
def PostVisualization(Sample):
	Col=Sample.columns.values
	Detector=np.where(Col=='Forecast')[0]
	
	if len(Detector)==2:
		Col[Detector[1]]='Forecast2'
		Sample.columns=Col
	
	Sample.to_json('CV_Result.json')

#	 Post - Visualization
#	Error=TstY-Forecast
#	plt.ion()
#	plt.figure(0)
#	plt.title('Histogram of Error')
#	plt.xlabel('Errors')
#	plt.ylabel('Frequency')
#	plt.hist(Error,50,histtype='bar',rwidth=1,alpha=0.75)
#	#plt.axis([-2,2,0,])
#	plt.grid(True)
#	plt.show()
#
#	A=pd.concat([TrnX,TrnY],axis=1)
#
#	plt.ion()
#	plt.figure(0)
#	plt.title('Pipe Pricing')
#	plt.xlabel('Diameter')
#	plt.ylabel('Price')
#	plt.plot(TrnX['length'],TrnY,'.',linewidth=2)
#	#plt.axis([-2,2,0,])
#	plt.grid(True)
#	plt.show()
#	bp()
#





#######################################################################
## Performance Measure [ Geleral Case ]
#######################################################################
def PerMeasure(X,Y):
	P=np.sqrt(np.mean(np.square(X - Y )))
	return P






#######################################################################
## Post Processing  [ General Case ]
#######################################################################
def PostProcessing(Prediction):
	global Param
	# Limitation
	NewPrediction=np.maximum(Prediction,Param['Min'])
	NewPrediction=np.minimum(NewPrediction,Param['Max'])
	return NewPrediction



#######################################################################
## Outlier Detector  [ General Case ]
#######################################################################
def OutlierDetector(TrainX,TrainY):
	# Outlier Detection
	NullDetector=TrainX.isnull().sum(axis=1)
	Good1=NullDetector[NullDetector == 0].index
	TrainX=TrainX.loc[Good1,:]
	TrainY=TrainY[Good1]
	return TrainX, TrainY








#######################################################################
## Optimal Alpha Detection
#######################################################################
def OptimalReg(TrainX,TrainY):
	global Param
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
			NormOutput=PostProcessing(Output)
			SubPerformance[CVindex]=PerMeasure(NormOutput,SubTestsY.values)

		Performance[Alphaindex]=SubPerformance.mean()

	BestReg=AlphaList[np.argmin(Performance)]
	print "                Best Regulator: ", BestReg, " Performance: ", np.min(Performance)
	return BestReg






#######################################################################
## Ridge Regerssion
#######################################################################
def RidgeReg(TrainX,TrainY):
	global Param

	BestReg=OptimalReg(TrainX,TrainY)	# Optimal Alpha Detection
	Reg=Ridge(alpha=BestReg,fit_intercept=True,normalize=False,copy_X=True)

	return Reg







#######################################################################
## Stochastic Gradient Descent
#####################################################################
def StochasticGradientDescent(TrainX,TrainY):
	global Param
	SGD = linear_model.SGDRegressor(loss="squared_loss")
	return SGD









#######################################################################
## GBM Internal Cross Validation
#####################################################################
def GBMHeldOutScore(CLF,TestsX,TestsY,NTry):

	Score=np.zeros(NTry)
	for i,y_pred in enumerate(CLF.staged_predict(TestsX)):
		Score[i]=PerMeasure(TestsY,y_pred)

	return Score







#######################################################################
## Gradient Boosting Training
#######################################################################
def GradientBoosting(TrainX,TrainY):
	global Param
	# Initial Training
	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,
	test_size=Param['GBM_CVP'])
	GBM = GradientBoostingRegressor(n_estimators=Param['GBM_NTry'],
		learning_rate=Param['GBM_LRate'],
		subsample=0.8,verbose=0)
	GBM.fit(SubTrainX,SubTrainY)
	

	# CV Performance Checking for the number of try
	CVscore  = GBMHeldOutScore(GBM,SubTestsX,SubTestsY,Param['GBM_NTry'])
	CVBestIter=np.argmin(CVscore)
	BestPerformance=np.min(CVscore)
	print CVBestIter,BestPerformance

	# Retraining
	GBM = GradientBoostingRegressor(n_estimators=CVBestIter,
		learning_rate=Param['GBM_LRate'],
		subsample=0.8,verbose=0)
	return GBM







#################################################################################
### KNN
#################################################################################
def KNearestNeighbors(TrainX , TrainY):
	global NClass, Param
	NNeighbors=Param['NNeighbors']
	KNN=KNeighborsRegressor(n_neighbors=NNeighbors,
		p=2,
		algorithm='auto',
		leaf_size=300,
		weights='uniform') # ,weights='distance'
	return KNN







#######################################################################
## XGBoost
#######################################################################
def XGBoost(TrainX,TrainY):
	global Param
	# Model Train
	xgtrain = xgb.DMatrix(TrainX, label=TrainY)
	XGB = xgb.train(Param['XGB'], xgtrain, Param['XGB_NR'])
	return XGB
	






#################################################################################
## Gaussian Process
#################################################################################
def GaussianPro(TrainX,TrainY):
	global Param
	GP = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
	return GP








#################################################################################
#	## Random Forest
#################################################################################
def RandomForest(TrainX,TrainY):
	global Param
	
	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['RF_CVP'])
	CandidateLeaf=range(2,3)
	Score=np.zeros(len(CandidateLeaf))
	for j in range(0,len(CandidateLeaf)):
		RF=RandomForestRegressor(n_estimators=50,verbose=0,
		min_samples_split=CandidateLeaf[j],
		min_samples_leaf=1,
		bootstrap=True,
		n_jobs=-1)
		RF.fit(SubTrainX,SubTrainY)
		
		# Refitting
		RF.fit(SubTrainX , SubTrainY)
		TestOutput=RF.predict(SubTestsX)
		Prediction=np.maximum(TestOutput,Param['Min'])
		Score[j]=PerMeasure(SubTestsY,Prediction)
	
	BestLeafIndex=np.argmin(Score)
	BestLeaf=CandidateLeaf[BestLeafIndex]
	BestPerformance=np.min(Score)
		
	print "BestLeaf: ", BestLeaf, " Best Performance: ", BestPerformance


	NTree=Param['RF_NTree']
	RF=RandomForestRegressor(n_estimators=NTree,verbose=0,min_samples_split=BestLeaf,
		min_samples_leaf=1,
		bootstrap=True,
		n_jobs=-1)

	return RF



#################################################################################
### Support Vector Machine Classification
#################################################################################
def SupportVectorMachine(TrainX,TrainY):
	global Param
	# CV Data Spliting
	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])
	
	# Cross validation to select the best C
	CandidateC=Param['C']
	Score=np.zeros(len(CandidateC))
	for i in range(0,len(CandidateC)):
		Support=SVR(C=CandidateC[i], kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=True,
			tol=CandidateC[i]/1000.0,
			cache_size=1000,
			verbose=0)
		Support.fit(SubTrainX,SubTrainY)
		TestOutput=Support.predict(SubTestsX)
		NewPrediction=PostProcessing(TestOutput)
		Score[i]=PerMeasure(NewPrediction,SubTestsY)
	
	BestCIndex=np.argmin(Score)
	BestC=CandidateC[BestCIndex]
	BestPerformance=np.min(Score)
		
	print "BestC: ", BestC, " Best Performance: ", BestPerformance

	# Final Prediction
	Support=SVR(C=BestC, kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=True,
		tol=CandidateC[i]/1000.0,
		cache_size=1000,
		verbose=0)
	return Support



################################################################################
## Main Machinenary Selection
################################################################################
def Machinery(MachineInfo,TrainX,TrainY,TestsX):
	global Param
	if MachineInfo=='Ridge':
		Attribute=RidgeReg(TrainX,TrainY)

	elif MachineInfo=='RF':
		Attribute=RandomForest(TrainX,TrainY)
	
	elif MachineInfo=='GBM':
		Attribute=GradientBoosting(TrainX,TrainY)

	elif MachineInfo=='NN':
		Temp=1

	elif MachineInfo=='SVM':
		Attribute=SupportVectorMachine(TrainX,TrainY)

	elif MachineInfo=='GP':
		Attribute=GaussianPro(TrainX,TrainY)

	elif MachineInfo=='XGB':
		Attribute=XGBoost(TrainX,TrainY)
		xgtest = xgb.DMatrix(TestsX)
		TestOutput = Attribute.predict(xgtest)
		return TestOutput

	elif MachineInfo=='SGD':
		Attribute=StochasticGradientDescent(TrainX,TrainY)


	elif MachineInfo=='KNN':
		Attribute=KNearestNeighbors(TrainX,TrainY)

	# PostProcessing
	Attribute.fit(TrainX,TrainY)
	TestOutput=Attribute.predict(TestsX)


	return TestOutput




######################################################################
## Ensemble Classification
#####################################################################
def Ensemble(Param, FunRMSE, SavePredict, TrainY):
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
		BestMachine=np.where(FunRMSE==np.min(FunRMSE))[1][0]
	
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
	NormForecast=PostProcessing(TempResult)
	PrintOutRMSE=PerMeasure(TrainY,NormForecast)
	print FunRMSE, PrintOutRMSE

	return NormForecast, Weight, BestGroup, BestMachine





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
def FinalTraining(ParamOriginal,TrainX,TrainY,TestsX,Weight,BestGroup,BestMachine):
	global Param
	Param=ParamOriginal
	MachineList=Param['MachineList']
	GroupDecision=Param['NGroup']

	SavePredict=np.zeros( ( len(BestMachine) , len(BestGroup) , len(TestsX) ) )
	for machindex in range(0,len(BestMachine)):
		Machine=MachineList[BestMachine[machindex]]

		for gt in range(0,len(BestGroup)):
			GD=GroupDecision[BestGroup[gt]]

			NG,TrainGroup,TestsGroup= GroupGen(GD,TrainX, TrainY, TestsX)
			GroupResult=np.zeros( len(TestsX))
			for grindex in range(0,NG):
			
				# Group Assigning
				TrnX=TrainX.iloc[TrainGroup[grindex],:]
				TrnY=TrainY.iloc[TrainGroup[grindex]]
				TstX=TestsX.iloc[TestsGroup[grindex],:]

				# Outlier Detection
				TrnX,TrnY=OutlierDetector(TrnX,TrnY)
				
				TrnX=TrnX.copy()
				TstX=TstX.copy()
				TrainNull=TrnX.isnull()
				TestsNull=TstX.isnull()
				for i in range(0,TrainNull.shape[1]):
					Temp=TrainNull.iloc[:,i].values
					TrnX.iloc[Temp,i]=-1
	
					Temp=TestsNull.iloc[:,i].values
					TstX.iloc[Temp,i]=-1

				# Preprocessing
				TrnX,TstX=Standardization(TrnX, TstX)
				
				# Forecast
				Forecast=Machinery(Machine,TrnX,TrnY,TstX)
				SavePredict[machindex,gt,TestsGroup[grindex]]=Forecast

	# Weighted Output Generation
	ReshapeSavePredict=np.reshape(SavePredict,(SavePredict.shape[0]*SavePredict.shape[1],
		SavePredict.shape[2]))
	TempResult= Weight * ReshapeSavePredict
	WeightedOutput=np.sum(TempResult,axis=0)
	WeightedOutput=PostProcessing(WeightedOutput)



	return WeightedOutput

















########################################################################
# Printing
########################################################################
def PostPrinting(WeightedOutput,FileName):
	# Prepare to print it out
	Converted=np.exp(WeightedOutput)-1
	Index=range(1,len(Converted)+1)
	Answer=pd.Series(Converted,index=Index,name='cost')
	Answer.to_csv(FileName,mode='w',index=True,index_label='id',header=True,float_format='%.5f')
	return Answer





