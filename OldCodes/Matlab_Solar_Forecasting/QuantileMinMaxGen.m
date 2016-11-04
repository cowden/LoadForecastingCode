function [QuantileMinMax,MinMaxMatrix]=QuantileMinMaxGen(CVForecast,ZonalTrainY,QuantileGroupSize)
%% Quantile Evaluation
QuantileMinMax=zeros(QuantileGroupSize,2);                  % Min / Max
MinMaxMatrix=zeros(size(ZonalTrainY,1),2);
[Group]=Wind_ReGroup(CVForecast,QuantileGroupSize);

for G=1:1:QuantileGroupSize
    SubTestY=ZonalTrainY(Group{G,1});
    
    if isempty(SubTestY)
        MIN=0;
        MAX=1;
    else
        
        MIN=min(SubTestY);
        MAX=max(SubTestY);
    end
    
    
    
    
    QuantileMinMax(G,1)=MIN;
    QuantileMinMax(G,2)=MAX;
    
    MinMaxMatrix(Group{G,1},1)=MIN;  % Min
    MinMaxMatrix(Group{G,1},2)=MAX;  % Max
end
