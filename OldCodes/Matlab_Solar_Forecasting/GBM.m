function [TestOutput,TrainOutput]=GBM(TrainX,TrainY,TestX,Parameters)


opts = [];
opts.loss = 'squaredloss'; % can be logloss or exploss
opts.subsamplingFactor = 0.5;
opts.randSeed = uint32(rand()*1000);
opts.shrinkageFactor = Parameters.shrinkageFactor;               % Bigger values go to over fitting, bigger values are slow. 0.01 is good It might be updating step amount
opts.maxTreeDepth = Parameters.maxTreeDepth;  % this was the default before customization
NumTree=Parameters.NumTree;



% Model
ModelGBM = SQBMatrixTrain(single(TrainX) , TrainY , uint32(NumTree),opts);

% Train Result
TrainOutput=SQBMatrixPredict( ModelGBM, single(TrainX) );

%% Forecasting
TestOutput=SQBMatrixPredict( ModelGBM, single(TestX) );
