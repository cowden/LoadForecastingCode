function [TestOutput,TrainOutput]=Ridge(TrainX,TrainY,TestX,~)




% No Group 12
% Lambda=[0.00005 0.0001 0.0005 0.0008 0.001 0.01 0.02 0.05  ...
%     0.1 0.5 0.8 0.9 1 1.5 2 3 4 5 5 10 15 20 25 30 40 50 55 60 65 70 80 90 100 110 120 130 140 150 200 210 300 400 500 1000];


Lambda=[0.1 0.5 1 2 3 4 5 ];
%Lambda=[0.001 0.01 0.02 0.05 0.1 0.5 0.8 0.9 1 1.5 2 3 4 5 10 15 20 25 30 40 50 55 60 65 70 80 90 100 110 120 130 140 150];

%%
CVNum=10;
CV=cvpartition(size(TrainX,1),'kfold',10);

RMSE=zeros(CVNum,length(Lambda));
for t=1:1:CVNum
    X1=TrainX(training(CV,t),:);
    Y1=TrainY(training(CV,t),:);
        
    X2=TrainX(test(CV,t),:);    
    Y2=TrainY(test(CV,t),:);
    
    b=ridge(Y1, X1, Lambda,0 );
    SubTestOutput=[ones(size(X2,1),1) X2]*b;
    
    E=bsxfun(@minus,SubTestOutput,Y2);
    RMSE(t,:)=sqrt(mean(E.^2));    
end

MeanRMSE=mean(RMSE,1);  %display(MeanRMSE);
[~,Loc]=min(MeanRMSE);
OptimalLambda=Lambda(min(Loc,length(Lambda)));display(OptimalLambda);

%% Zero Mean
Model=ridge(TrainY, TrainX , OptimalLambda, 0 ); % , 'Distribution','normal'); %,'Distribution','normal','Link','probit');

%% Train Output
TrainOutput=[ones(size(TrainX,1),1) TrainX]*Model;

%% Forecasting
TestOutput=[ones(size(TestX,1),1) TestX]*Model;







