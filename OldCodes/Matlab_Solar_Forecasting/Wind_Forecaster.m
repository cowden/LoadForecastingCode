function [TestOutput,RMSE]=Wind_Forecaster(TrainX,TrainY,TestX,ZonalTrainY,FunName,Command)

%%
[TestOutput,TrainOutput]=feval(str2func(FunName{1}),TrainX,TrainY,TestX,Command);

%% Post-Processing
Fitted=smooth(TrainOutput,Command.OutputSmooth);
Fitted=max(Fitted,min(ZonalTrainY));
Fitted=min(Fitted,max(ZonalTrainY));

RMSE=sqrt(mean( (Fitted-TrainY).^2) );

