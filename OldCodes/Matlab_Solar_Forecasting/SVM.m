function [TestOutput,TrainOutput]=SVM(TrainX,TrainY,TestX,Command)

%
%
CostSet=Command.CostSet;
LossSet=Command.LossSet;
InterSet=Command.InterSet;

SVMBase='-s 3 -t 2 -d 3 ';
SVMTail=' -h 0';


CV=cvpartition(size(TrainX,1),'holdout',0.2);
Y1=TrainY(training(CV),:);
Y2=TrainY(test(CV),:);
X1=TrainX(training(CV),:);
X2=TrainX(test(CV),:);


%% Decide the Optimal Cost
InitialTol=0.002;
InitialLoss=0.04;

RMSE=zeros(length(CostSet),1);
for i=1:1:length(CostSet)
    SVMCase=[SVMBase, ' -c ', num2str(CostSet(i)), ' -e ', num2str(InitialTol) ,' -p ' , num2str(InitialLoss) , SVMTail];
    
    %% Traininv
    Model=svmtrain(Y1,X1,SVMCase);
    [TE, ~,~]=svmpredict(Y2,X2,Model);
    
    RMSE(i)=sqrt( mean ( (TE-Y2).^2) );
end
[~,Loc]=min(RMSE);
FinalCost=CostSet(Loc);




%% Decide the Loss
RMSE=zeros(length(LossSet),1);
for i=1:1:length(LossSet)
    SVMCase=[SVMBase, ' -c ', num2str(FinalCost) , ' -e ',num2str(InitialTol) ,' -p ' , num2str(LossSet(i)) , SVMTail];
    
    %% Traininv
    Model=svmtrain(Y1,X1,SVMCase);
    [TE, ~,~]=svmpredict(Y2,X2,Model);
    
    RMSE(i)=sqrt( mean ( (TE-Y2).^2) );
end
[~,Loc]=min(RMSE);
FinalLoss=LossSet(Loc);




%% Decide the Tolerance
RMSE=zeros(length(InterSet),1);
for i=1:1:length(InterSet)
    SVMCase=[SVMBase, ' -c ', num2str(FinalCost) , ' -e ', num2str(InterSet(i)) , ' -p ' , num2str(FinalLoss) , SVMTail];
    
    %% Traininv
    Model=svmtrain(Y1,X1,SVMCase);
    [TE, ~,~]=svmpredict(Y2,X2,Model);
    
    RMSE(i)=sqrt( mean ( (TE-Y2).^2) );
end
[~,Loc]=min(RMSE);
FinalInter=InterSet(Loc);



%% Final Cost One more time
RMSE=zeros(length(CostSet),1);
for i=1:1:length(CostSet)
    SVMCase=[SVMBase, ' -c ', num2str(CostSet(i)), ' -e ', num2str(FinalInter) ,' -p ' , num2str(FinalLoss) , SVMTail];
    
    %% Traininv
    Model=svmtrain(Y1,X1,SVMCase);
    [TE, ~,~]=svmpredict(Y2,X2,Model);
    
    RMSE(i)=sqrt( mean ( (TE-Y2).^2) );
end
[~,Loc]=min(RMSE);
FinalCost=CostSet(Loc);

% %}
% 
% FinalCost=0.21;
% FinalInter=0.001;
% FinalLoss=0.04;






%% Final Training
SVMCase=[SVMBase, ' -c ', num2str(FinalCost), ' -e ', num2str(FinalInter) ,' -p ', num2str(FinalLoss) , SVMTail];
display(SVMCase);
Model=svmtrain(TrainY,TrainX,SVMCase);
[TrainOutput, ~,~]=svmpredict(TrainY,TrainX,Model);
[TestOutput, ~,~]=svmpredict((1:1:size(TestX,1))',TestX,Model);




