function [Quantile]=QuantileGeneratorNew(Forecast,Parameter,QuantileMode)


%%
LTestX=length(Forecast);
Quantile=zeros(LTestX,99);

%%
switch QuantileMode    
    case 'Uniform'              
        for m=1:1:LTestX
            for i=1:1:99
                Segment= Parameter(m)*(i-50);
                Quantile(m,i)=Forecast(m)+Segment;
            end
        end

    case 'Norm'        
        for m=1:1:LTestX        
            RND=normrnd(Forecast(m),Parameter(m),[10000,1]);
            Quantile(m,:)=quantile(RND,99);
        end
        
    case 'Laplace'
        for m=1:1:LTestX
            b=sqrt(Parameter(m)/2);
            Seed=rand(10000,1)-0.5;
            RND=Forecast(m)-b*sign(Seed).*log(1-2*abs(Seed));
            Quantile(m,:)=quantile(RND,99);
        end        
        
    case 'Gamma'
        for m=1:1:LTestX
            b1=sqrt(Parameter(m,1)/2);   % Left
            b2=sqrt(Parameter(m,2)/2);   % Right
            Seed=rand(10000,1)-0.5;            
            
            RND1=Forecast(m)-b1*sign(Seed).*log(1-2*abs(Seed));
            RND2=Forecast(m)-b2*sign(Seed).*log(1-2*abs(Seed));
            
            Quantile(m,1:49)=quantile(RND1,(0.01:0.01:0.49)');
            Quantile(m,50:99)=quantile(RND2,(0.5:0.01:0.99)');
        end
end





       
