function [FinalQuantile]=UnifiedQuantile(ReducedCVForecast,ReducedZonalTrainY,FinalForecast,QuantileMinMax,QuantileGroupSize,GroupForFinalFore,QuantileMode)

[GroupForCVFore]=Wind_ReGroup(ReducedCVForecast,QuantileGroupSize);

%% Quantile Evaluation
FinalQuantile=zeros(length(FinalForecast),99);
for G=1:1:QuantileGroupSize
    SubCVForecast=ReducedCVForecast(GroupForCVFore{G,1});       
    SubTestY=ReducedZonalTrainY(GroupForCVFore{G,1});   
    MIN=QuantileMinMax(G,1);
    MAX=QuantileMinMax(G,2);
    [Parameter,~]=QuantileEvaluation(SubTestY,SubCVForecast,MIN,MAX,QuantileMode);    
    display(Parameter);
    
    SubFinalForecast=FinalForecast(GroupForFinalFore{G,1});
    [Quantile]=QuantileGenerator(SubFinalForecast,Parameter,MIN,MAX,QuantileMode);
    FinalQuantile(GroupForFinalFore{G,1},:)=Quantile;
end



function [Quantile]=QuantileGenerator(Forecast,Parameter,MIN,MAX,QuantileMode)


LTestX=length(Forecast);

switch QuantileMode    
    case 'Same'
        Quantile=repmat(Forecast,1,99);
        
    case 'Norm'        
        Quantile=zeros(LTestX,99);
        for i=1:1:LTestX
            RND=normrnd(Forecast(i),Parameter(1),[10000,1]);
            Quantile(i,:)=quantile(RND,99);
        end
        
    case 'Uniform'         
        Quantile=zeros(LTestX,99);
        for i=1:1:99
            Segment= Parameter(1)*(i-50);
            Quantile(:,i)=Forecast+Segment;
        end
        
    case 'Laplace'
        b=sqrt(Parameter(1)/2);
        Seed=rand(10000,1)-0.5;
        Quantile=zeros(LTestX,99);
        for i=1:1:LTestX
            RND=Forecast(i)-b*sign(Seed).*log(1-2*abs(Seed));
            Quantile(i,:)=quantile(RND,99);
        end
        
        
    case 'Gamma'
        b1=sqrt(Parameter(1)/2);   % Left
        b2=sqrt(Parameter(2)/2);   % Right
        Seed=rand(10000,1)-0.5;
        Quantile=zeros(LTestX,99);
        for i=1:1:LTestX
            RND1=Forecast(i)-b1*sign(Seed).*log(1-2*abs(Seed));
            RND2=Forecast(i)-b2*sign(Seed).*log(1-2*abs(Seed));
            
            Quantile(i,1:49)=quantile(RND1,(0.01:0.01:0.49)');
            Quantile(i,50:99)=quantile(RND2,(0.5:0.01:0.99)');
        end
       
end
Quantile=min(Quantile,MAX);
Quantile=max(Quantile,MIN);

