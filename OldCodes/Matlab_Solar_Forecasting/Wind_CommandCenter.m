function [Command]=Wind_CommandCenter(Order,Z)



%% First Case
if Order==1
    InputType=[11];
    DaySum=[1];
    FunGroup={'Bagging','SVM','GBM','RandomForest','NeuralNetA'}; % 'Ridge','SVM','GBM','RandomForest','NeuralNetA','Gauss', Bagging
    
    %% NN
    NNSize=3500;                      % < 100
    NNTestRatio=0.2;
    NNValNum=20;
    NNFunction='trainscg';
   
    %% GP
    SampleRatio=0.3;
    Iter=100;
    
    
    %% GBM
    NumTree = 4000;  % 2000 might be better
    opts.shrinkageFactor = 0.03;               % Bigger values go to over fitting, bigger values are slow. 0.01 is good It might be updating step amount
    opts.maxTreeDepth = uint32(10);  % this was the default before customization
    
    %% GBM ToolBox
    BagSize=500;              % 300
    LR=0.01;
    LeafRatio=5;
    
    %% RF
    RFTree=500;
    Mtry=3;    % 3
    Nodesize=7;
    %
    %     Command.CostSet=[0.21];
    %     Command.InterSet=[0.001];
    %     Command.LossSet=[0.04];    % 0.04
    
    Command.CostSet=[0.18 0.20 0.22];
    Command.InterSet=[0.001 0.003 0.005];
    Command.LossSet=[0.01 0.03 0.05];    % 0.04
    
    
elseif Order==2
    InputType=[8];
    DaySum=[0];
    GroupType=[1];
    FunGroup={'Ridge'};
    
    %% NN
    NNSize=1200;                      % < 100
    NNTestRatio=0.2;
    NNValNum=20;
    NNFunction='trainscg';
    
    %% GP
    SampleRatio=0.2;
    SF = 1;
    ell = 0.5;
    sn = 18;
    
    %% GBM
    NumTree = 3000;  % 2000 might be better
    opts.shrinkageFactor = 0.01;               % Bigger values go to over fitting, bigger values are slow. 0.01 is good It might be updating step amount
    opts.maxTreeDepth = uint32(10);  % this was the default before customization
    
    %% GBM ToolBox
    BoosTSize=3000;              % 300
    LR=0.01;
    LeafRatio=5;
    
    %% RF
    RFTree=200;
    Mtry=3;    % 3
    Nodesize=7;
    %
    %     Command.CostSet=[0.21];
    %     Command.InterSet=[0.001];
    %     Command.LossSet=[0.04];    % 0.04
    
    Command.CostSet=[0.16 0.18 0.20 0.22 0.24];
    Command.InterSet=[0.0005 0.001 0.003 0.005 0.007];
    Command.LossSet=[0.005 0.01 0.03 0.05 0.07];    % 0.04
    
elseif Order==3              % Good   % Good for Group 2 4 Good
    InputType=[5 6 7];
    DaySum=[0 1];
    DayBand=1;
    GroupType=[1 2 3 4];
    FunGroup={'Boosting'};
    
    %% NN
    NNSize=1200;                      % < 100
    NNTestRatio=0.2;
    NNValNum=20;
    NNFunction='trainscg';
    
    %% GP
    SampleRatio=0.2;
    SF = 1;
    ell = 0.5;
    sn = 18;
    
    %% GBM
    NumTree = 4000;  % 2000 might be better
    opts.shrinkageFactor = 0.008;               % Bigger values go to over fitting, bigger values are slow. 0.01 is good It might be updating step amount
    opts.maxTreeDepth = uint32(5);  % this was the default before customization
    
    %% GBM ToolBox
    BoosTSize=3000;              % 300
    LR=0.01;
    LeafRatio=15;
    
    %% Ridge Regression
    Lambda=[0.001 0.01 0.02 0.05 0.1 0.2 0.5 0.8 1 1.2 1.5 2 3 5 10 20 30];
    
    %% SVM
    % SVM='-s 3 -t 2 -d 2 -c 0.2 -e 0.01 -p 0.1 -h 0';
    SVM='-s 3 -t 2 -d 2 -c 0.150 -e 0.005 -p 0.050 -h 0';
    % c: Cost
    % e: tolerance to terminate             0.005 or 0.01
    % p: epsilon in loss function           0.05 0.03 0.01
    % h: Shrinking 0 or 1
    
    %% RF
    RFTree=1000;
    Mtry=4;
    Nodesize=7;
    
elseif Order==4
    InputType=[5 6 7];
    DaySum=[1 2];
    Pre=1;
    Post=1;
    GroupType=[1 2 3 4];
    FunGroup={'RandomForest'};
    
    %% NN
    NNSize=1200;                      % < 100
    NNTestRatio=0.2;
    NNValNum=20;
    NNFunction='trainscg';
    
    %% GP
    SampleRatio=0.2;
    SF = 1;
    ell = 0.5;
    sn = 18;
    
    %% GBM
    NumTree = 4000;  % 2000 might be better
    opts.shrinkageFactor = 0.008;               % Bigger values go to over fitting, bigger values are slow. 0.01 is good It might be updating step amount
    opts.maxTreeDepth = uint32(5);  % this was the default before customization
    
    %% GBM ToolBox
    BoosTSize=3000;              % 300
    LR=0.01;
    
    %% Ridge Regression
    Lambda=[0.001 0.01 0.02 0.05 0.1 0.2 0.5 0.8 1 1.2 1.5 2 3 5 10 20 30];
    
    %% SVM
    % SVM='-s 3 -t 2 -d 2 -c 0.2 -e 0.01 -p 0.1 -h 0';
    SVM='-s 3 -t 2 -d 2 -c 0.150 -e 0.005 -p 0.050 -h 0';
    % c: Cost
    % e: tolerance to terminate             0.005 or 0.01
    % p: epsilon in loss function           0.05 0.03 0.01
    % h: Shrinking 0 or 1
    
    %% RF
    RFTree=2000;
    Mtry=4;
    Nodesize=3;
    
elseif Order==5                         % Also Very good for group 4
    InputType=[5 6 7];
    DaySum=[1 2];
    Pre=1;
    Post=1;
    GroupType=[1 2 3 4];
    FunGroup={'RandomForest'};
    
    %% NN
    NNSize=1200;                      % < 100
    NNTestRatio=0.2;
    NNValNum=20;
    NNFunction='trainscg';
    
    %% GP
    SampleRatio=0.3;
    SF = 1;
    ell = 0.5;
    sn = 18;
    
    %% GBM
    NumTree = 4000;  % 2000 might be better
    opts.shrinkageFactor = 0.008;               % Bigger values go to over fitting, bigger values are slow. 0.01 is good It might be updating step amount
    opts.maxTreeDepth = uint32(5);  % this was the default before customization
    
    %% GBM ToolBox
    BoosTSize=3000;              % 300
    LR=0.01;
    
    %% Ridge Regression
    Lambda=[0.001 0.01 0.02 0.05 0.1 0.2 0.5 0.8 1 1.2 1.5 2 3 5 10 20 30];
    
    %% SVM
    % SVM='-s 3 -t 2 -d 2 -c 0.2 -e 0.01 -p 0.1 -h 0';
    SVM='-s 3 -t 2 -d 2 -c 0.150 -e 0.005 -p 0.050 -h 0';
    % c: Cost
    % e: tolerance to terminate             0.005 or 0.01
    % p: epsilon in loss function           0.05 0.03 0.01
    % h: Shrinking 0 or 1
    
    %% RF
    RFTree=3000;
    Mtry=4;
    Nodesize=3;
          
end







%% Saving Basic Data Structure Setting
Command.InputType=InputType;
Command.DaySum=DaySum;
Command.FunGroup=FunGroup;


%% NN
Command.NNSize=NNSize;                      % < 100
Command.NNTestRatio=NNTestRatio;
Command.NNValNum=NNValNum;
Command.NNFunction=NNFunction;

%% GP
Command.SampleRatio=SampleRatio;
Command.Iter = Iter;

%% GBM
Command.NumTree = NumTree;  % 2000 might be better
Command.shrinkageFactor = opts.shrinkageFactor;   % Bigger values go to over fitting, bigger values are slow. 0.01 is good It might be updating step amount
Command.maxTreeDepth = opts.maxTreeDepth;

%% GBM ToolBox
Command.BagSize=BagSize;              % 300
Command.LR=LR;
Command.LeafRatio=LeafRatio;

%% RF
Command.RFTree=RFTree;
Command.Mtry=Mtry;
Command.Nodesize=Nodesize;






if Z==1
    Command.InputSmooth=3;
elseif Z==2
    Command.InputSmooth=3;   % 5
elseif Z==3
    Command.InputSmooth=3;
elseif Z==4
    Command.InputSmooth=3;
elseif Z==5
    Command.InputSmooth=3;
elseif Z==6
    Command.InputSmooth=3;
elseif Z==7
    Command.InputSmooth=3;
elseif Z==8
    Command.InputSmooth=3;
elseif Z==9
    Command.InputSmooth=3;
elseif Z==10
    Command.InputSmooth=3;
end

%
%     %% SVM
%     % SVM='-s 3 -t 2 -d 2 -c 0.2 -e 0.01 -p 0.1 -h 0';
%     SVM='-s 3 -t 2 -d 2 -c 0.150 -e 0.005 -p 0.050 -h 0';
%     % c: Cost
%     % e: tolerance to terminate             0.005 or 0.01
%     % p: epsilon in loss function           0.05 0.03 0.01
%     % h: Shrinking 0 or 1

