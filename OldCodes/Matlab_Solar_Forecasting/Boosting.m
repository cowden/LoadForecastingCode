function [TestOutput,TrainOutput]=Boosting(TrainX,TrainY,TestX,Parameters)


CV=cvpartition(size(TrainX,1),'holdout',0.2);

LR=Parameters.LR;
BoosTSize=Parameters.BoosTSize;   % Increase Much
LeafNumber=round(size(TrainX,1)/Parameters.LeafRatio);
display(LeafNumber);

% RMSE=zeros(CVNum,1);
% for t=1:1:CVNum

NPred=1; % round(size(TrainX,2)*Parameters.NPred);

X1=TrainX(training(CV),:);
Y1=TrainY(training(CV),:);

X2=TrainX(test(CV),:);
Y2=TrainY(test(CV),:);

Tree=RegressionTree.template('MinLeaf',LeafNumber); % 'Surrogate','off','Prune','on','MergeLeaves','on',
Model=fitensemble(X1 , Y1 ,'LSBoost',BoosTSize,Tree,'LearnRate',LR,'nprint',100);%,'resample','on','replace','off','fresample',1);

        %    'resample','on','replace','on','fresample',0.5); % 'NPredToSample',NPred,


%% Error Analysis
Errors=loss(Model,X2,Y2,'mode','cumulative');
[Val,Loc]=min(Errors);
display(Val);display(Loc);

figure(7);
plot(Errors,'color','r');hold on;
plot(resubLoss(Model,'mode','cumulative'),'color','b');hold off;
xlim([100 BoosTSize]);
title(Val);
grid on;

    
    
Tree=RegressionTree.template('MinLeaf',LeafNumber); % 'Surrogate','off','Prune','on','MergeLeaves','on',
Model=fitensemble(TrainX , TrainY ,'LSBoost',Loc,Tree,'LearnRate',LR,'nprint',100);%,'resample','on','replace','off','fresample',1);
%  ,'resample','on','replace','on','fresample',0.5); % ,'NPredToSample',NPred,

    
%     keyboard;
%     
% end

% 
% 
% %% CV
% Cvpart=cvpartition(TrainY,'holdout',0.2);
% X1 = TrainX(training(Cvpart),:);
% Y1 = TrainY(training(Cvpart),1);
% X2 = TrainX(test(Cvpart),:);
% Y2 = TrainY(test(Cvpart),1);

% 
% 
% 
% %% Model
% Tree=RegressionTree.template('Surrogate','on','MinLeaf',20);
% CVModel=fitensemble(X1 , Y1 , 'LSBoost',3000,'tree','LearnRate',0.3,'nprint',100);
% 
% 
% figure;
% plot(loss(CVModel,X2,Y2,'mode','cumulative'));hold on;
% plot(kfoldLoss(CVModel,'mode','cumulative'),'r.');hold off;
% 






%% Model
%Tree=RegressionTree.template('Surrogate','on','MinLeaf',5);
%Model = fitensemble(D1,TrainY_D,'LSBoost',BoosTSize,'tree','LearnRate',0.5,'resample','on');
% Model=fitensemble(TrainX , TrainY , 'LSBoost',BoosTSize,'tree','LearnRate',LR,'nprint',1000);
%Model = regularize(Model,'lambda',0.01,'verbose',1); % Regularization using Lasso
%Model = shrink(Model,'weightcolumn',1);

% figure(10);
% plot(resubLoss(CVModel,'mode','cumulative'));
% grid on;


%% Train
TrainOutput=predict(Model,TrainX);



%% Forecasting
TestOutput=predict(Model,TestX);









