function [FeelSoGood]=OutlierDitection(TempTrainX,ZonalTrainY,FunName,Command)

% %
% OutlierLimit=0.80; 
% 
% %%
% [~,TrainOutput]=feval(str2func(FunName),TempTrainX,ZonalTrainY,TempTrainX,Command);
% Residual=abs(TrainOutput-ZonalTrainY);
% 
% [~,ID]=sort(abs(Residual),'ascend');
% 
% 
% FeelSoBad=ID(round(length(ZonalTrainY)*OutlierLimit):end);
% FeelSoGood=setxor( (1:1:length(ZonalTrainY))',FeelSoBad);
% FeelSoGood=sort(FeelSoGood);
% 


FeelSoGood=1:1:length(ZonalTrainY);
