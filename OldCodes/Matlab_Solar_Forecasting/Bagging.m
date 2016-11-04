function [TestOutput,TrainOutput]=Bagging(TrainX,TrainY,TestX,Parameters)


BoosTSize=Parameters.BagSize;   % Increase Much

t = RegressionTree.template('minleaf',2);
%Tree=RegressionTree.template('MinLeaf',LeafNumber); % 'Surrogate','off','Prune','on','MergeLeaves','on',
Model=fitensemble(TrainX , TrainY ,'Bag',BoosTSize,t,'type','regression','nprint',100);%,'resample','on','replace','off','fresample',1);
%  ,'resample','on','replace','on','fresample',0.5); % ,'NPredToSample',NPred,



%% Train
TrainOutput=predict(Model,TrainX);



%% Forecasting
TestOutput=predict(Model,TestX);






