#######################################################################
## Import Packages
#######################################################################
# Basic Packages
import sys
from sys import path
import pandas as pd
import numpy as np
import os
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pdb, traceback
import datetime
import time


# Cross Validation Packages
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn import linear_model

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error


# Figure
import matplotlib.pyplot as plt


# Our Own Code
import CasePrep.GEF2014_Helper as Helper









#######################################################################
## Standardization [ General Case ]
#######################################################################
def Standardization ( TrainX, TestsX ):

    # Depending on whether there is TestsX or not
    try:
        MegaX=TrainX.append(TestsX,ignore_index=True)
    except:
        MegaX=TrainX.copy()

    # Determine the Unique
    STD=np.std(MegaX,axis=0)
    NonZero=np.nonzero(STD)[0]
    Unique=MegaX.iloc[:,NonZero]

    # Normalization
    Normal=(Unique - Unique.mean()) / Unique.std()

    # Set data
    TrainX=Normal.iloc[0:TrainX.shape[0],:]
    TestsX=Normal.iloc[TrainX.shape[0]:,:]

    return TrainX, TestsX



##########################################################################
### Null Detect Function [ General Case ]
########################################################################
def NullDetector(TrnX,TstX):
    # Null value Detection
    TrnX2=TrnX.copy()
    TstX2=TstX.copy()
    TrainNull=TrnX2.isnull()
    TestsNull=TstX2.isnull()

    for i in range(0,TrainNull.shape[1]):
        Temp=TrainNull.iloc[:,i].values
        TrnX2.iloc[Temp,i]=-1

        Temp=TestsNull.iloc[:,i].values
        TstX2.iloc[Temp,i]=-1


    return TrnX2,TstX2



##########################################################################
### Cross Validation Function [ General Case ]
########################################################################
def CrossVal(Param,TrainX,TrainY):

    CVMethod=Param['CVMethod']
    MachineList=Param['MachineList']

    ## Cross Validation
    kf = KFold( len(TrainY) ,shuffle=True, n_folds=Param['CVN'] )


    ## Play the Game
    SavePredict=np.zeros( ( len(MachineList) , len(TrainY) ) )
    FunRMSE=np.zeros( len(MachineList) )
    for machindex in range(0,len(MachineList)):
        Machine=MachineList[machindex]

        cc=0
        for train_index, test_index in kf:
            cc=cc+1
            SubTrainX=TrainX.iloc[train_index,:]
            SubTrainY=TrainY.iloc[train_index]
            SubTestsX=TrainX.iloc[test_index,:]
            SubTestsY=TrainY.iloc[test_index]

            if CVMethod=="Random":
                SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['CVP'])


            # Re-Indexing
            SubTrainX.index=np.arange(0,len(SubTrainX))
            SubTrainY.index=np.arange(0,len(SubTrainY))
            SubTestsX.index=np.arange(0,len(SubTestsX))
            SubTestsY.index=np.arange(0,len(SubTestsY))

            # Outlier Detection
            SubTrainX,SubTrainY=Helper.OutlierDetector(SubTrainX,SubTrainY)

            # Null value Detection
            SubTrainX,SubTestsX=NullDetector(SubTrainX,SubTestsX)


            # Standardization
            SubTrainX,SubTestsX=Standardization(SubTrainX, SubTestsX)

            # Forecast
            Forecast=Machinery(Param,Machine,SubTrainX,SubTrainY,SubTestsX,SubTestsY)
            SavePredict[machindex, test_index]=Forecast
            NormForecast=PostProcessing(Forecast,Param)
            print Machine, "CV:", cc, PerMeasure(SubTestsY,NormForecast)

        SingleMachineOutput=SavePredict[machindex, :]
        TempResult=np.mean(SingleMachineOutput,axis=0)
        NormForecast=PostProcessing(TempResult,Param)
        print Machine, PerMeasure(TrainY,NormForecast)

    TempResult=np.mean(SavePredict,axis=0)
    NormForecast=PostProcessing(TempResult,Param)
    print FunRMSE, PerMeasure(TrainY,NormForecast)


    # Visualization
    Forecast=pd.DataFrame(NormForecast,columns=['Forecast'])
    Error=pd.DataFrame((TrainY.values-NormForecast)**2,columns=['Error'])
    Sample=pd.concat([TrainX,TrainY,Forecast,Error],axis=1)

    # Print Out
    PostVisualization(Sample)


    return FunRMSE, SavePredict



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






########################################################################
### Performance Measure [ Geleral Case ]
########################################################################
def PerMeasure(X,Y):
    # RMSE
    P=np.sqrt(np.mean(np.square(X - Y )))
    return P





########################################################################
### Post Processing  [ General Case ]
########################################################################
def PostProcessing(Prediction, Param):
    # Limitation
    NewPrediction=np.maximum(Prediction,Param['Min'])
    NewPrediction=np.minimum(NewPrediction,Param['Max'])
    return NewPrediction
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




########################################################################
### Ridge Regerssion
########################################################################
def RidgeReg(Param,TrainX,TrainY):

    ## Define the list of regulating parameters
    AlphaList=Param['Reg_Reg']

    ## Define the ridge regression estimator
    RIDGE=Ridge(alpha=AlphaList[0],fit_intercept=True,normalize=False,copy_X=True)

    ## Grid Seaerching Parameter Generation
    Param_grid = {'alpha': AlphaList}

    ## Run the Cross Validation
    CV=GridSearchCV(estimator=RIDGE, n_jobs=-1, param_grid=Param_grid, cv= Param['Reg_CVN'])
    CV.fit(TrainX,TrainY)
    print(CV.best_params_)

    ## Build the final Ridge Estimator
    Reg=Ridge(alpha=CV.best_params_["alpha"],fit_intercept=True,normalize=False,copy_X=True)

    return Reg





########################################################################
### Stochastic Gradient Descent
######################################################################
#def StochasticGradientDescent(TrainX,TrainY):
#	global Param
#	SGD = linear_model.SGDRegressor(loss="squared_loss")
#	return SGD
#
#
#
#
#
#
#
#
#
########################################################################
### GBM Internal Cross Validation
######################################################################
#def GBMHeldOutScore(CLF,TestsX,TestsY,NTry):
#
#	Score=np.zeros(NTry)
#	for i,y_pred in enumerate(CLF.staged_predict(TestsX)):
#		Score[i]=PerMeasure(TestsY,y_pred)
#
#	return Score
#
#
#
#
#
#
#
########################################################################
### Gradient Boosting Training
########################################################################
#def GradientBoosting(TrainX,TrainY):
#	global Param
#	# Initial Training
#	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,
#	test_size=Param['GBM_CVP'])
#	GBM = GradientBoostingRegressor(n_estimators=Param['GBM_NTry'],
#		learning_rate=Param['GBM_LRate'],
#		subsample=0.8,verbose=0)
#	GBM.fit(SubTrainX,SubTrainY)
#
#
#	# CV Performance Checking for the number of try
#	CVscore  = GBMHeldOutScore(GBM,SubTestsX,SubTestsY,Param['GBM_NTry'])
#	CVBestIter=np.argmin(CVscore)
#	BestPerformance=np.min(CVscore)
#	print CVBestIter,BestPerformance
#
#	# Retraining
#	GBM = GradientBoostingRegressor(n_estimators=CVBestIter,
#		learning_rate=Param['GBM_LRate'],
#		subsample=0.8,verbose=0)
#	return GBM
#
#
#
#
#
#
#
##################################################################################
#### KNN
##################################################################################
#def KNearestNeighbors(TrainX , TrainY):
#	global NClass, Param
#	NNeighbors=Param['NNeighbors']
#	KNN=KNeighborsRegressor(n_neighbors=NNeighbors,
#		p=2,
#		algorithm='auto',
#		leaf_size=300,
#		weights='uniform') # ,weights='distance'
#	return KNN
#
#
#
#
#
#
#
########################################################################
### XGBoost
########################################################################
#def XGBoost(TrainX,TrainY):
#	global Param
#	# Model Train
#	xgtrain = xgb.DMatrix(TrainX, label=TrainY)
#	XGB = xgb.train(Param['XGB'], xgtrain, Param['XGB_NR'])
#	return XGB
#
#
#
#
#
#
#
##################################################################################
### Gaussian Process
##################################################################################
#def GaussianPro(TrainX,TrainY):
#	global Param
#	GP = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#	return GP
#
#
#
#
#
#
#
#
##################################################################################
##	## Random Forest
##################################################################################
def RandomForest(Param,TrainX,TrainY):

    ## Estimator Definition
    RF=RandomForestRegressor(n_estimators=100,verbose=0,
                             max_featuers="sqrt",
                             min_samples_split=2,
                             min_samples_leaf=1,
                             bootstrap=True,
                             warm_start=False,
                             n_jobs=-1)

    ## Grid Seaerch
    Param_grid = {'n_estimators': Param['RF_NTree'],
                  'max_features': Param['RF_Feature'],
                  'min_samples_split': Param['RF_Split'],
                  'min_samples_leaf':Param['RF_Leaf']
                  }
    ParamGrid=list(ParameterGrid(Param_grid))

    ## Run the Cross Validation
    CV=GridSearchCV(estimator=RF, n_jobs=-1, param_grid=ParamGrid, cv= Param['RF_CVN'])
    CV.fit(TrainX,TrainY)
    print(CV.best_params_)

    ## Define the Final Estimator
    RF=RandomForestRegressor(n_estimators=CV.best_params_['n_estimators'],
                             verbose=0,
                             max_features=CV.best_params_['max_featuers'],
                             min_samples_split=CV.best_params_['min_samples_split'],
                             min_samples_leaf=CV.best_params_['min_samples_leaf'],
                             bootstrap=True,
                             n_jobs=-1)

    return RF


#	for j in range(0,len(CandidateLeaf)):
#
#		RF.fit(SubTrainX,SubTrainY)
#
#		# Refitting
#		RF.fit(SubTrainX , SubTrainY)
#		TestOutput=RF.predict(SubTestsX)
#		Prediction=np.maximum(TestOutput,Param['Min'])
#		Score[j]=PerMeasure(SubTestsY,Prediction)
#
#	BestLeafIndex=np.argmin(Score)
#	BestLeaf=CandidateLeaf[BestLeafIndex]
#	BestPerformance=np.min(Score)
#
#	print "BestLeaf: ", BestLeaf, " Best Performance: ", BestPerformance




#
#
#
##################################################################################
#### Support Vector Machine Classification
##################################################################################
#def SupportVectorMachine(TrainX,TrainY):
#	global Param
#	# CV Data Spliting
#	SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=Param['SVM_CVP'])
#
#	# Cross validation to select the best C
#	CandidateC=Param['C']
#	Score=np.zeros(len(CandidateC))
#	for i in range(0,len(CandidateC)):
#		Support=SVR(C=CandidateC[i], kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=True,
#			tol=CandidateC[i]/1000.0,
#			cache_size=1000,
#			verbose=0)
#		Support.fit(SubTrainX,SubTrainY)
#		TestOutput=Support.predict(SubTestsX)
#		NewPrediction=PostProcessing(TestOutput)
#		Score[i]=PerMeasure(NewPrediction,SubTestsY)
#
#	BestCIndex=np.argmin(Score)
#	BestC=CandidateC[BestCIndex]
#	BestPerformance=np.min(Score)
#
#	print "BestC: ", BestC, " Best Performance: ", BestPerformance
#
#	# Final Prediction
#	Support=SVR(C=BestC, kernel='rbf', degree=3, gamma=Param['Gamma'] , shrinking=True,
#		tol=CandidateC[i]/1000.0,
#		cache_size=1000,
#		verbose=0)
#	return Support
#
#
#
#################################################################################
### Main Machinenary Selection
#################################################################################
def Machinery(Param,MachineInfo,TrainX,TrainY,TestsX,TestsY):
    if MachineInfo=='Ridge':
        Attribute=RidgeReg(Param,TrainX,TrainY)

    elif MachineInfo=='RF':
        Attribute=RandomForest(Param,TrainX,TrainY)

    elif MachineInfo=='GBM':
        Temp=1
#        Attribute=GradientBoosting(TrainX,TrainY)

    elif MachineInfo=='NN':
        Temp=1

    elif MachineInfo=='SVM':
        Temp=1
 #       Attribute=SupportVectorMachine(TrainX,TrainY)

    elif MachineInfo=='GP':
        Temp=1
#        Attribute=GaussianPro(TrainX,TrainY)

    elif MachineInfo=='XGB':
        Temp=1
 #       Attribute=XGBoost(TrainX,TrainY)
 #       xgtest = xgb.DMatrix(TestsX)
 #       TestOutput = Attribute.predict(xgtest)
 #       return TestOutput

    elif MachineInfo=='SGD':
        Temp=1
#        Attribute=StochasticGradientDescent(TrainX,TrainY)

    elif MachineInfo=='KNN':
        Temp=1
#		Attribute=KNearestNeighbors(TrainX,TrainY)

    # PostProcessing
    Attribute.fit(TrainX,TrainY)
    TestOutput=Attribute.predict(TestsX)


    return TestOutput







#######################################################################
### Ensemble Classification
######################################################################
#def Ensemble(Param, FunRMSE, SavePredict, TrainY):
#	EnsemOption=Param['Ensem']
#	ReshapeFunRMSE=np.reshape(FunRMSE,FunRMSE.size)
#	ReshapeSavePredict=np.reshape(SavePredict,(FunRMSE.size,SavePredict.shape[2]))
#
#	if EnsemOption=='Best':
#		# Weight
#		Weight=np.array([1])
#
#		# Result Making
#		TempResult=ReshapeSavePredict[np.argmin(ReshapeFunRMSE)]
#
#		# Best Index
#		BestGroup=np.where(FunRMSE==np.min(FunRMSE))[0][0]
#		BestMachine=np.where(FunRMSE==np.min(FunRMSE))[1][0]
#
#	elif EnsemOption=='Simple':
#		# Weight
#		Weight=np.ones(FunRMSE.size)*1.0/FunRMSE.size
#		Weight=Weight[np.newaxis].transpose()
#
#		# Result Making
#		TempResult=np.mean(ReshapeSavePredict,axis=0)
#
#		# Best Index
#		BestGroup=range(0,FunRMSE.shape[1])
#		BestMachine=range(0,FunRMSE.shape[0])
#
#	elif EnsemOption=='Weight':
#		# Weight
#		InverseWeight=1.0 / ReshapeFunRMSE
#		Weight=InverseWeight / sum(InverseWeight)
#		Weight=Weight[np.newaxis].transpose()
#
#		# Result Making
#		TempResult= Weight * ReshapeSavePredict
#		TempResult=np.sum(TempResult,axis=0)
#
#		# Best Index
#		BestGroup=range(0,FunRMSE.shape[1])
#		BestMachine=range(0,FunRMSE.shape[0])
#
#	# Performance Measure
#	NormForecast=PostProcessing(TempResult)
#	PrintOutRMSE=PerMeasure(TrainY,NormForecast)
#	print FunRMSE, PrintOutRMSE
#
#	return NormForecast, Weight, BestGroup, BestMachine
#
#
#
#
#
##################################################################################
### Second Wave
###################################################################################
#def SecondWave(MeanPredict,WeightedOutput,TrainX,TestsX):
#
#	Forecast=pd.DataFrame(MeanPredict,columns=['Forecast'])
#	WeightedOutput=pd.DataFrame(WeightedOutput,columns=['Forecast'])
#
#	NewTrainX=pd.concat([TrainX,Forecast],axis=1)
#	NewTestsX=pd.concat([TestsX,WeightedOutput],axis=1)
#
#	return NewTrainX, NewTestsX
#
#
#
#
#
#####################################################################################
### Final Training
###################################################################################
#def FinalTraining(ParamOriginal,TrainX,TrainY,TestsX,Weight,BestGroup,BestMachine):
#	global Param
#	Param=ParamOriginal
#	MachineList=Param['MachineList']
#	GroupDecision=Param['NGroup']
#
#	SavePredict=np.zeros( ( len(BestMachine) , len(BestGroup) , len(TestsX) ) )
#	for machindex in range(0,len(BestMachine)):
#		Machine=MachineList[BestMachine[machindex]]
#
#		for gt in range(0,len(BestGroup)):
#			GD=GroupDecision[BestGroup[gt]]
#
#			NG,TrainGroup,TestsGroup= GroupGen(GD,TrainX, TrainY, TestsX)
#			GroupResult=np.zeros( len(TestsX))
#			for grindex in range(0,NG):
#
#				# Group Assigning
#				TrnX=TrainX.iloc[TrainGroup[grindex],:]
#				TrnY=TrainY.iloc[TrainGroup[grindex]]
#				TstX=TestsX.iloc[TestsGroup[grindex],:]
#
#				# Outlier Detection
#				TrnX,TrnY=OutlierDetector(TrnX,TrnY)
#
#				TrnX=TrnX.copy()
#				TstX=TstX.copy()
#				TrainNull=TrnX.isnull()
#				TestsNull=TstX.isnull()
#				for i in range(0,TrainNull.shape[1]):
#					Temp=TrainNull.iloc[:,i].values
#					TrnX.iloc[Temp,i]=-1
#
#					Temp=TestsNull.iloc[:,i].values
#					TstX.iloc[Temp,i]=-1
#
#				# Preprocessing
#				TrnX,TstX=Standardization(TrnX, TstX)
#
#				# Forecast
#				Forecast=Machinery(Machine,TrnX,TrnY,TstX)
#				SavePredict[machindex,gt,TestsGroup[grindex]]=Forecast
#
#	# Weighted Output Generation
#	ReshapeSavePredict=np.reshape(SavePredict,(SavePredict.shape[0]*SavePredict.shape[1],
#		SavePredict.shape[2]))
#	TempResult= Weight * ReshapeSavePredict
#	WeightedOutput=np.sum(TempResult,axis=0)
#	WeightedOutput=PostProcessing(WeightedOutput)
#
#
#
#	return WeightedOutput
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
#
#########################################################################
## Printing
#########################################################################
#def PostPrinting(WeightedOutput,FileName):
#	# Prepare to print it out
#	Converted=np.exp(WeightedOutput)-1
#	Index=range(1,len(Converted)+1)
#	Answer=pd.Series(Converted,index=Index,name='cost')
#	Answer.to_csv(FileName,mode='w',index=True,index_label='id',header=True,float_format='%.5f')
#	return Answer
#
#








########################################################################
### Optimal Alpha Detection
########################################################################
#def OptimalReg(TrainX,TrainY,Param):
#    CVN=Param['Reg_CVN']
#    CVP=Param['Reg_CVP']
#    AlphaList=Param['Reg_Reg']
#
#    Performance=np.zeros(len(AlphaList))
#    for Alphaindex in range(0,len(AlphaList)):
#        Regulator=AlphaList[Alphaindex]
#
#        SubPerformance=np.zeros(CVN)
#        for CVindex in range(0,CVN):
#            # Set indice for subtrainx and subtestx to perform the CV
#            SubTrainX,SubTestsX,SubTrainY,SubTestsY=train_test_split(TrainX,TrainY,test_size=CVP)
#
#            f=Ridge(alpha=Regulator,fit_intercept=True,normalize=False,copy_X=True)
#            f.fit(SubTrainX,SubTrainY)
#
#            Output=f.predict(SubTestsX)
#            NormOutput=PostProcessing(Output)
#            SubPerformance[CVindex]=PerMeasure(NormOutput,SubTestsY.values)
#
#        Performance[Alphaindex]=SubPerformance.mean()
#
#	BestReg=AlphaList[np.argmin(Performance)]
#	print "                Best Regulator: ", BestReg, " Performance: ", np.min(Performance)
#	return BestReg

