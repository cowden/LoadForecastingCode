function [TestOutput,TrainOutput]=NeuralNetA(TrainX,TrainY,TestX,Command)

MeanY=mean(TrainY);
ZeroMeanY=TrainY-MeanY;


TNum=10;
TrainSave=zeros(size(TrainX,1),TNum);
TestSave=zeros(size(TestX,1),TNum);
for t=1:1:TNum
    
    %
    P=TrainX';
    T=ZeroMeanY';
    
    %% Data Ratio
    TrainRatio=1-Command.NNTestRatio;
    ValRatio=Command.NNTestRatio;
    TestRatio=0;
    
    %% Setting the NN
    Number=ceil(Command.NNSize/size(TrainX,2));
    Number=round(Number);
    net = feedforwardnet(Number); % 40 is good
    
    %% initialization
    net.initFcn='initlay';
    net.layers{1}.transferFcn='tansig';  % tansig
    net.layers{1}.initFcn='initnw';
    net.layers{2}.transferFcn='purelin';
    net.layers{2}.initFcn='initnw';
    
    %% Training Mode
    % traincgp trainscg trainlm
    net.trainFcn='trainscg';  %'trainbr'  trainscg  trainlm traingdx trainrp traingdm trainbfg traincgp traincgb
    net.divideMode = 'sample';  % sample none
    net.divideFcn='dividerand';  %'divideblock'; dividerand divideind 'divideint',
    net.performFcn= 'msereg';
    
    %% Training, Varlidation, Test
    net.divideParam.trainRatio=TrainRatio;
    net.divideParam.valRatio=ValRatio;
    net.divideParam.testRatio=TestRatio;
    %net.performParam.ratio = 0.5;
    
    net.trainParam.epochs = 3000;
    net.trainParam.time=10*60;      % Second
    net.trainParam.goal =0.001;
    net.trainParam.max_fail=Command.NNValNum;
    net.trainParam.mu = 1;
    % net.trainParam.mu_dec=0.8; % 0.8
    % net.trainParam.mu_inc= 1.5;
    % net.trainParam.mu_max = 1e10;
    % net.trainParam.show=Command.NNValNum;
    % net.trainParam.showWindow=1;
    % net.trainParam.min_grad=1e-10;
    %net.trainParam.sigma=5e-5;
    %net.trainParam.lambda=5e-7;
    % net.trainParam.minstep=0.0001;
    
    %% Training
    net = configure(net,P,T);
    [net,~] = train(net,P,T);
    
    
    
    
    
    %% Train
    TrainOutput= sim(net,TrainX')'+MeanY;
    TrainSave(:,t)=TrainOutput;
    
    
    %% Forecasting
    TestOutput= sim(net,TestX')'+MeanY;
    TestSave(:,t)=TestOutput;
    
end
TrainOutput=mean(TrainSave,2);
TestOutput=mean(TestSave,2);





