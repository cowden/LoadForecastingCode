function [TestOutput,TrainOutput]=RandomForest(TrainXOrgn,TrainYOrgn,TestX,Command)

% No Group 12


LocalOutlierMode=0;

if LocalOutlierMode==1
    [FeelSoGood]=OutlierDitection(TrainXOrgn,TrainYOrgn);
else
    FeelSoGood=(1:1:size(TrainYOrgn,1))';
end
TrainX=TrainXOrgn(FeelSoGood,:);
TrainY=TrainYOrgn(FeelSoGood);


%% Zero Mean
MeanY=mean(TrainY);
ZeroMeanY=TrainY-MeanY;


%% Options
RFTree=Command.RFTree;
Mtry=round(size(TrainX,2)/Command.Mtry);
extra_options.nodesize = Command.Nodesize;
extra_options.do_trace = 0;
extra_options.sampsize = round(size(TrainXOrgn,1)*2/3); %2/3
extra_options.replace =1 ;
extra_options.importance = 0;
extra_options.localImp = 0;
extra_options.proximity = 0;
extra_options.oob_prox = 0;
extra_options.keep_inbag = 0;
extra_options.nPerm = 0;  % 3


% 
% %% CV Setting
% CV=cvpartition(size(TrainX,1),'holdout',0.2);
% Y1=TrainY(training(CV),:);
% MeanY1=mean(Y1);
% ZeroY1=Y1-MeanY1;
% Y2=TrainY(test(CV),:);
% X1=TrainX(training(CV),:);
% X2=TrainX(test(CV),:);
% 
% %% RMSE Calculation
% RMSE=zeros(5,1);
% for i=1:1:5
%     Add=-8+(i-1)*4;
%     Model = regRF_train(X1,ZeroY1, 100, Mtry+Add,extra_options);
%     TrainOutput = regRF_predict(X2,Model)+MeanY1;
%     RMSE(i)=sqrt( mean ( (TrainOutput-Y2).^2));
% end
% [~,Loc]=min(RMSE);
% OptimalAdd=-8+(Loc-1)*4;
% display(Mtry+OptimalAdd);



OptimalAdd=0;



%% Training
% extra_options.replace = 0 ;
Model = regRF_train(TrainX,ZeroMeanY, RFTree, Mtry+OptimalAdd,extra_options);


%%
TrainOutput = regRF_predict(TrainXOrgn,Model)+MeanY;
TestOutput = regRF_predict(TestX,Model)+MeanY;

%
%     figure;
%     plot(TrainY,'color','b');hold on;
%     plot(TrainOutput,'color','r');hold off;
%     grid on;







