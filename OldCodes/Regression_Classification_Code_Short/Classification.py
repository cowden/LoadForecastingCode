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

# Debugging
from pdb import set_trace as bp
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split


# Classification Packages.
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import gaussian_process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC

## Case Dependent Packages
from Crime import ParameterSetting, GroupGen







#######################################################################
## Data Loading
#######################################################################
def DataLoading(DataPath):
	FileList=os.listdir(DataPath)
	FileList=FileList[1:]
	
	DataSaver=[0] * len(FileList)
	for file in range( 0,len(FileList) ):
		FileName=FileList[file]
		DataSaver[file]=pd.read_csv(DataPath+FileName,na_values=-1)

	return DataSaver , FileList







#########################################################################
## Cross Validation Function
#######################################################################
def CrossVal(ParamOriginal,TrainX,TrainY):
	global Param , NClass

	Param=ParamOriginal
	MachineList=Param['MachineList']
	CVN=Param['CVN']
	NClass=Param['NC']
	GroupDecision=Param['NGroup']

	## Cross Validation
	kf = KFold( len(TrainY) ,shuffle=True, n_folds=CVN )

	## Play the Game
	SavePredict=np.zeros( ( len(MachineList) , len(GroupDecision) , len(TrainY), NClass ) )
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


				GroupResult=np.zeros( (len(SubTestsX) , NClass) )
				for grindex in range(0,NG):
					# Group Assigning
					TrnX=SubTrainX.iloc[TrainGroup[grindex],:]
					TrnY=SubTrainY.iloc[TrainGroup[grindex]]
					TstX=SubTestsX.iloc[TestsGroup[grindex],:]
					TstY=SubTestsY.iloc[TestsGroup[grindex]]

					# Outlier Detection
					#TrnX,TrnY=OutlierDetector(TrnX,TrnY)

					# Preprocessing
					TrnX,TstX=Standardization(TrnX, TstX)

					# Forecast
					Forecast=Machinery(Machine,TrnX,TrnY,TstX)
					GroupResult[TestsGroup[grindex],:]=Forecast
					NormForecast=PostProcessing(Forecast)
					TempPerformance=PerMeasure(TstY,NormForecast)
					print Machine, "Main Group:", GD, "Sub Group:", grindex, "CV:", c, TempPerformance


				SavePredict[machindex, gt, test_index,:]=GroupResult
				NormForecast=PostProcessing(GroupResult)
				TempPerformance=PerMeasure(SubTestsY,NormForecast)
				print Machine, "Main Group:", GD, "CV:", c, TempPerformance


			SingleOutput=SavePredict[machindex, gt, :,:]
			NormForecast=PostProcessing(SingleOutput)
			TempPerformance=PerMeasure(TrainY,NormForecast)
			FunRMSE[machindex,gt]=TempPerformance
			print Machine, "Main Group:", GD, TempPerformance


		SingleOutput=SavePredict[machindex, :, :,:]
		TempResult=np.mean(SingleOutput,axis=0)
		NormForecast=PostProcessing(TempResult)
		TempPerformance=PerMeasure(TrainY,NormForecast)
		print Machine, TempPerformance

	TempResult=np.mean(SavePredict,axis=0)
	TempResult=np.mean(TempResult,axis=0)
	NormForecast=PostProcessing(TempResult)
	Forecast=pd.DataFrame(NormForecast,columns=range(0,NClass))
	Max=pd.DataFrame(np.argmax(NormForecast,axis=1) , columns=['Max'])
	Sample=pd.concat([TrainX,TrainY,Forecast,Max],axis=1)

	# Print Out
	PostVisualization(Sample)
	print FunRMSE, PerMeasure(TrainY,NormForecast)

	return FunRMSE, SavePredict




#######################################################################
## Outlier Detection
#######################################################################
def OutlierDetector(TrainX,TrainY):
	global NClass, Param
	
	# Outlier Detection
	STrnX, STstX = Standardization(TrainX,[])
	TempPrediction=Machinery('LogReg',STrnX,TrainY,STrnX)

	# Finding
	Prediction=np.log(np.maximum(np.minimum(TempPrediction,1-1e-15),1e-15))
	Solution=np.zeros( (Forecast.shape[0],NClass) )
	WeightedOutput=TrainY.astype(int)
	Solution[np.arange(0,Solution.shape[0]),WeightedOutput]=1
	Performance=-1.0/Solution.shape[0]*np.sum(Solution*Prediction, axis=1)

	SortedIndex=np.argsort(Performance)[::-1]
	
	Bad=SortedIndex[0 : np.round(len(SortedIndex)*0.05)]
	Good=SortedIndex[np.round(len(SortedIndex)*0.05) : ]

	TrnX=TrainX.iloc[Good,:]
	TrnY=TrainY.iloc[Good]

	return TrnX, TrnY




#########################################################################
### Class Performance Evaluation
########################################################################
def PerMeasure(Answer,Prediction):
	global NClass
	Prediction=np.log(np.maximum(np.minimum(Prediction,1-1e-15),1e-15))
	Solution=np.zeros( (Answer.shape[0],NClass) )
	WeightedOutput=Answer.astype(int)
	Solution[np.arange(0,Solution.shape[0]),WeightedOutput]=1
	Performance=-1.0/Solution.shape[0]*sum(sum(Solution*Prediction))
	return 	Performance







#######################################################################
## Standardization
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







#######################################################################
## Visualization
#######################################################################
def PreVisualization(TrainX, TrainY, NClass, UniqueTerm):

	## PColor Figure
	AFactor='DayOfWeek'
	BFactor='Month'
	CFactor='Hour'
	
	# JSON file Saving
	Pivot=TrainX.pivot_table(index=AFactor,columns=BFactor,values=CFactor,aggfunc=len)
	NormalPivot=Pivot / sum(sum(Pivot.values))
	NormalPivot.to_json('Pivot.json')
	
	# Plotting
	x,y=np.mgrid[0:8,0:13]
	plt.ion()
	plt.figure(0)
	plt.pcolor(x,y,NormalPivot.values,cmap='cool')
	plt.axis([x.min(),x.max(),y.min(),y.max()])
	plt.colorbar()
	labels=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
	plt.xticks(np.arange(0.5,7), labels, rotation='vertical')
	plt.yticks(np.arange(0.5,12), np.arange(1,13))
	plt.title('Color plot for the Pivot Table')
	plt.xlabel(AFactor)
	plt.ylabel(BFactor)
	plt.show()


	## Donuts Plotting
	Numbers=np.zeros(NClass)
	for index in range(0,NClass):
		Numbers[index]=TrainY[TrainY==index].shape[0]
	Numbers=Numbers.astype(int)


#	A=pd.Series(UniqueTerm,name='Class')
#	B=pd.Series(Numbers,name='Frequency')
#	C=pd.concat([A,B],axis=1)
#	C.to_csv('Frequency.csv',mode='w',index=False,header=True)

	Sorted=np.argsort(Numbers)[::-1]
	TopNumber=Numbers[Sorted[:10]]
	TopNumber=np.append(TopNumber, sum(Numbers)-sum(TopNumber) )
	TopLabel=np.append(UniqueTerm[Sorted[:10]],"Others")


	Data=pd.Series(TopNumber, index=TopLabel)
	Data.to_json('Frequency.json')



	# Plot the pie chart
	plt.ion()
	plt.figure(1)
	plt.pie(TopNumber , labels=TopLabel,shadow=True,startangle=90)
	plt.axis('equal')
	plt.title('Number of cases per class')
	plt.show()



	## Feature Importance
	TempFeature=np.zeros( (10,TrainX.shape[1]))
	STrainX, STestX = Standardization ( TrainX, [] )
	for i in range(0,1):
		SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(STrainX,TrainY,test_size=0.1)
		RF=RandomForestClassifier(n_estimators=100,verbose=1,
			min_samples_split=7,
			min_samples_leaf=1,
			bootstrap=True,
			class_weight='auto',n_jobs=-1)
		RF.fit(SubTestsX,SubTestsY)
		TempFeature[i,:]=RF.feature_importances_


	## Feature Importance Plotting
		FeatureImportance=np.mean(TempFeature,axis=0)
		FeatureError=np.std(TempFeature,axis=0)
	if len(FeatureImportance) > 15 :
		Sorted=np.argsort(FeatureImportance)[::-1]
		TopFeature=FeatureImportance[Sorted[:15]]
		TopFeature=TopFeature / sum(TopFeature)
		TopLabel=TrainX.columns[Sorted[:15]]
		TopError=FeatureError[:15]

	elif len(FeatureImportance) <= 15:
		TopFeature=FeatureImportance
		TopLabel=TrainX.columns
		TopError=FeatureError

#	A=pd.Series(TrainX.columns,name='Features')
#	B=pd.Series(FeatureImportance,name='Importance')
#	C=pd.concat([A,B],axis=1)
#	C.to_csv('CrimeFeatureImportance.csv',mode='w',index=False,header=True)


	Data=pd.Series(FeatureImportance, index=TrainX.columns)
	Data.to_json('CrimeFeatureImportance.json')

	# Feature Importanace Plotting
	Y_Pos=np.arange(len(TopFeature))
	plt.ion()
	plt.figure(2)
	plt.barh(Y_Pos,TopFeature,xerr=TopError,align='center',alpha=0.4)  # TopError
	plt.yticks(Y_Pos,TopLabel)
	plt.xlabel('Importance')
	plt.title('Feature Importance')
	plt.show()







#######################################################################
## Post Visualization
#######################################################################
def PostVisualization(Sample):
	Sample2=Sample









########################################################################
### Optimal Alpha Detection
########################################################################
def OptimalReg(TrainX,TrainY):
	global NClass, Param
	
	CVN=Param['Log_CVN']
	CVP=Param['Log_CVP']

	AlphaList=Param['Log_Reg']
	Performance=np.zeros(len(AlphaList))
	for Alphaindex in range(0,len(AlphaList)):
		Alpha=AlphaList[Alphaindex]

		SubPerformance=np.zeros(CVN)
		for CVindex in range(0,CVN):
			# Set indice for subtrainx and subtestx to perform the CV
			SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=CVP)

			Logistic=LogisticRegression(C=Alpha,solver='lbfgs')   # ,fit_intercept=True
			Logistic.fit(SubTrainX,SubTrainY)

			Output=Logistic.predict_proba(SubTestsX)
			NormPrediction=Assigning(Output,Logistic)
			Prediction=PostProcessing(NormPrediction)
			SubPerformance[CVindex]=PerMeasure(SubTestsY,Prediction)
	
		Performance[Alphaindex]=SubPerformance.mean()

	BestAlpha=AlphaList[np.argmin(Performance)]
	print "                   Best Alpha: ", BestAlpha, " Performance: ", np.min(Performance)
	return BestAlpha







########################################################################
### Logistic Regerssion
#######################################################################
def LogisticReg(TrainX,TrainY):
	global NClass,Param

	# Tic=time.time()
	# Find the Best Regulization Parameter
	BestReg=OptimalReg(TrainX,TrainY)

	# Training the Ridge Model
	Logistic=LogisticRegression(C=BestReg,verbose=0,solver='lbfgs')   # ,fit_intercept=True
	# multi_class='multinomial',

	return Logistic









#######################################################################
## GBM Internal Cross Validation
#####################################################################
def GBMHeldOutScore(CLF,TestsX,TestsY,NTry):
	global NClass

	Score=np.zeros(NTry)
	for i,y_pred in enumerate(CLF.staged_predict_proba(TestsX)):

		Prediction=np.zeros( (y_pred.shape[0],NClass))
		for j in range(0 , len(CLF.classes_) ):
			Prediction[:,CLF.classes_[j]]=y_pred[:,j]

		Score[i]=PerMeasure(TestsY,Prediction)

	return Score









#######################################################################
## Gradient Boosting Training
#######################################################################
def GradientBoosting(TrainX,TrainY):
	global NClass, Param

	# Initial Training
	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,
	test_size=Param['GBM_CVP'])
	GBM = GradientBoostingClassifier(n_estimators=Param['GBM_NTry'],
		learning_rate=Param['GBM_LRate'],
		subsample=0.5,verbose=1)
	GBM.fit(SubTrainX,SubTrainY)

	# CV Performance Checking for the number of try
	CVscore  = GBMHeldOutScore(GBM,SubTestsX,SubTestsY,Param['GBM_NTry'])
	CVBestIter=np.argmin(CVscore)
	BestPerformance=np.min(CVscore)
	print CVBestIter,BestPerformance

	# Retraining
	GBM = GradientBoostingClassifier(n_estimators=(CVBestIter),
		learning_rate=Param['GBM_LRate'],
		subsample=0.5,verbose=1)
	return GBM









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
#	## Random Forest
#################################################################################
def RandomForest(TrainX,TrainY):
	global NClass, Param
	
#	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['RF_CVP'])
#	CandidateLeaf=range(6,8)
#	Score=np.zeros(len(CandidateLeaf))
#	for j in range(0,len(CandidateLeaf)):
#		RF=RandomForestClassifier(n_estimators=100,verbose=1,
#		min_samples_split=CandidateLeaf[j],
#		min_samples_leaf=1,
#		bootstrap=True,
#		class_weight='auto',n_jobs=-1)
#		RF.fit(SubTrainX,SubTrainY)
#		TestOutput=RF.predict_proba(SubTestsX)
#		Prediction=PostProcessing(TestOutput,RF)
#		Score[j]=PerMeasure(SubTestsY,Prediction)
#	
#	BestLeafIndex=np.argmin(Score)
#	BestLeaf=CandidateLeaf[BestLeafIndex]
#	BestPerformance=np.min(Score)
#		
#	print "BestLeaf: ", BestLeaf, " Best Performance: ", BestPerformance



	NTree=Param['RF_NTree']
	RF=RandomForestClassifier(n_estimators=NTree,verbose=0,
		min_samples_split=7,
		min_samples_leaf=1,
		bootstrap=True,
		class_weight='auto'
		,n_jobs=-1)

	return RF





#################################################################################
### KNN
#################################################################################
def KNearestNeighbors(TrainX , TrainY):
	global NClass, Param
	NNeighbors=Param['NNeighbors']
	KNN=KNeighborsClassifier(n_neighbors=NNeighbors,
		p=2,
		algorithm='auto',
		leaf_size=2,
		weights='uniform') # ,weights='distance'
	
	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])
	KNN.fit(SubTrainX,SubTrainY)
	TestOutput=KNN.predict_proba(SubTestsX)
	Prediction=PostProcessing(TestOutput,KNN)
	Score=PerMeasure(SubTestsY,Prediction)
	print "             KNN Performance: " , Score
	return KNN










#################################################################################
### Nu Support Vector Machine Classification
#################################################################################
def NuSVMClass(TrainX,TrainY):
	global NClass, Param

	# CV Data Spliting
	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])
	
	# Cross validation to select the best C
	CandidateNu=Param['Nu']
	Score=np.zeros(len(CandidateNu))
	for i in range(0,len(CandidateNu)):
		NuSupport=NuSVC(nu=CandidateNu[i], kernel='rbf', degree=3, gamma=0.5 , shrinking=False,
			probability=True,
			tol=0.001,
			cache_size=1000,
			verbose=0)
		NuSupport.fit(SubTrainX,SubTrainY)
		TestOutput=NuSupport.predict_proba(SubTestsX)
		Prediction=PostProcessing(TestOutput,NuSupport)

		Score[i]=PerMeasure(SubTestsY,Prediction)
	BestCIndex=np.argmin(Score)
	BestNu=CandidateNu[BestCIndex]
	BestPerformance=np.min(Score)
		
	print "BestNu: ", BestNu, " Best Performance: ", BestPerformance

	# Final Prediction
	NuSupport=NuSVC(nu=BestNu, kernel='rbf', degree=3, gamma=0.5 , shrinking=False,
		probability=True,
		tol=0.001,
		cache_size=1000,
		verbose=0)
	return NuSupport









#################################################################################
### Support Vector Machine Classification
#################################################################################
def SupportVectorClass(TrainX,TrainY):
	global NClass, Param
	
	# CV Data Spliting
	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])
	
	# Cross validation to select the best C
	CandidateC=Param['C']
	Score=np.zeros(len(CandidateC))
	for i in range(0,len(CandidateC)):
		Support=SVC(C=CandidateC[i], kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=False,
			probability=True,
			tol=0.001,
			cache_size=1000,
			verbose=0)
		Support.fit(SubTrainX,SubTrainY)
		TestOutput=Support.predict_proba(SubTestsX)
		Prediction=PostProcessing(TestOutput,NuSupport)
		Score[i]=PerMeasure(SubTestsY,Prediction)
	
	BestCIndex=np.argmin(Score)
	BestC=CandidateC[BestCIndex]
	BestPerformance=np.min(Score)
		
	print "BestC: ", BestC, " Best Performance: ", BestPerformance

	# Final Prediction
	Support=SVC(C=BestC, kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=False,
		probability=True,
		tol=0.001,
		cache_size=1000,
		verbose=0)
	return Support










################################################################################
## Main Machinenary Selection
################################################################################
def Machinery(MachineInfo,TrainX,TrainY,TestsX):
	global NClass, Param
	
	# Input data information
	if MachineInfo=='LogReg':
		Attribute=LogisticReg(TrainX,TrainY)

	elif MachineInfo=='RF':
		Attribute=RandomForest(TrainX,TrainY)

	elif MachineInfo=='GBM':
		Attribute=GradientBoosting(TrainX,TrainY)

	elif MachineInfo=='NuSVM':
		Attribute=NuSVMClass(TrainX,TrainY)

	elif MachineInfo=='SVM':
		Attribute=SupportVectorClass(TrainX,TrainY)

	elif MachineInfo=='XGB':
		Attribute=XGBoost(TrainX,TrainY)
		xgtest = xgb.DMatrix(TestsX)
		TestOutput = Attribute.predict(xgtest)
		return TestOutput

	elif MachineInfo=='KNN':
		Attribute=KNearestNeighbors(TrainX,TrainY)
	
	# PostProcessing
	Attribute.fit(TrainX,TrainY)
	TestOutput=Attribute.predict_proba(TestsX)
	NormPrediction=Assigning(TestOutput,Attribute)

	return NormPrediction







######################################################################
## Assigninging Function
#####################################################################
def Assigning(TestOutput,Attribute):
	global NClass
	Prediction=np.zeros( (len(TestOutput) , NClass))
	for i in range(0,len(Attribute.classes_)):
		Prediction[:,Attribute.classes_[i]]=TestOutput[:,i]
	return Prediction








######################################################################
## Post Processing: Class Assigning and Normalization
#####################################################################
def PostProcessing(TestOutput):
	global NClass, Param
	# Limitation and Normalization
	NewPrediction=np.maximum(TestOutput,Param['Min'])
	Sum=NewPrediction.sum(axis=1)
	NormPrediction=NewPrediction / Sum[:,np.newaxis]

	return NormPrediction










######################################################################
## Ensemble Classification
#####################################################################
def Ensemble(Param, FunRMSE, SavePredict, TrainY):
	EnsemOption=Param['Ensem']

	if EnsemOption=='Best':
		# Weight
		Weight=np.array([1])
		
		# Best Index
		BestGroup=np.where(FunRMSE==np.min(FunRMSE))[0][0]
		BestMachine=np.where(FunRMSE==np.min(FunRMSE))[1][0]
	
		# Result Making
		TempResult=SavePredict[BestMachine, BestGroup, : , : ]

	elif EnsemOption=='Simple':
		# Weight
		Weight=np.ones( (FunRMSE.shape[0],FunRMSE.shape[1]))*1.0/FunRMSE.size
		
		# Best Index
		BestGroup=range(0,FunRMSE.shape[1])
		BestMachine=range(0,FunRMSE.shape[0])
	
		# Result Making
		TempResult=np.mean(SavePredict,axis=0)
		TempResult=np.mean(TempResult,axis=0)

		# Find the weights for the weighted averaging
	elif EnsemOption=='Weight':
		# Weight
		InverseWeight=1.0 / FunRMSE
		Weight=InverseWeight / sum(sum(InverseWeight))

		# Result Making
		TempResult=np.zeros( (SavePredict.shape[2],SavePredict.shape[3]) )
		for m in range(0,SavePredict.shape[0]):
			for g in range(0,SavePredict.shape[1]):
				TempResult=TempResult+Weight[m,g] * SavePredict[m,g,:,:]

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
	
	Forecast=pd.DataFrame(MeanPredict,columns=range(0,MeanPredict.shape[1]))
	WeightedOutput=pd.DataFrame(WeightedOutput,columns=range(0,MeanPredict.shape[1]))
	
	NewTrainX=pd.concat([TrainX,Forecast],axis=1)
	NewTestsX=pd.concat([TestsX,WeightedOutput],axis=1)

	return NewTrainX, NewTestsX









####################################################################################
## Final Training
##################################################################################
## Final Training: All training data is used to train the model
def FinalTraining(ParamOriginal,TrainX,TrainY,TestsX,Weight,BestGroup,BestMachine):
	global NClass, Param
	Param=ParamOriginal
	MachineList=Param['MachineList']
	CVN=Param['CVN']
	NClass=Param['NC']
	GroupDecision=Param['NGroup']

	SavePredict=np.zeros( (len(BestMachine), len(BestGroup), len(TestsX) , NClass) )
	for machindex in range(0,len(BestMachine)):
		Machine=MachineList[BestMachine[machindex]]

		for gt in range(0,len(BestGroup)):
			GD=GroupDecision[BestGroup[gt]]

			NG,TrainGroup,TestsGroup= GroupGen(GD,TrainX, TrainY, TestsX)
			GroupResult=np.zeros((len(TestsX),NClass))
			for grindex in range(0,NG):
			
				# Group Assigning
				TrnX=TrainX.iloc[TrainGroup[grindex],:]
				TrnY=TrainY.iloc[TrainGroup[grindex]]
				TstX=TestsX.iloc[TestsGroup[grindex],:]

				# Outlier Detection
				#TrnX,TrnY=OutlierDetector(TrnX,TrnY)
			
				# Preprocessing
				TrnX,TstX=Standardization(TrnX, TstX)
				
				# Forecast
				Forecast=Machinery(Machine,TrnX,TrnY,TstX)
				SavePredict[machindex,gt,TestsGroup[grindex],:]=Forecast

	# Weighted Output Generation
	ReshapeSavePredict=np.reshape(SavePredict,(SavePredict.shape[0]*SavePredict.shape[1],
		SavePredict.shape[2],SavePredict.shape[3]))
		
	# Result Making
	WeightedOutput=np.zeros( (SavePredict.shape[2],SavePredict.shape[3]) )
	for m in range(0,SavePredict.shape[0]):
		for g in range(0,SavePredict.shape[1]):
			WeightedOutput=WeightedOutput+Weight[m,g] * SavePredict[m,g,:,:]

	NormPrediction=PostProcessing(WeightedOutput)


	return NormPrediction

















####################################################################################
### Post Printing
####################################################################################
def PostPrinting(WeightedOutput,UniqueTerm,FileName):
	Answer=pd.DataFrame(WeightedOutput,columns=UniqueTerm)
	Answer=Answer.sort(axis=1)
	Answer.to_csv(FileName,mode='w',index=True,index_label='Id',header=True,float_format='%.5f')









