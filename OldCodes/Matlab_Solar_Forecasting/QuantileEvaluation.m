function [Parameter,Val]=QuantileEvaluation(TestY,Forecast,MIN,MAX,QuantileMode)


switch QuantileMode
    case 'Uniform'
        
        DeCoeffStep=0.000025;
        StepNumber=500;
        
        QP=zeros(StepNumber,1);
        for t=1:StepNumber
            DeCoeff=DeCoeffStep*t;
            TestY_D2=TestY;
            
            Quantile=zeros(length(Forecast),99);
            for i=1:1:99
                Segment= DeCoeff*(i-50);
                Quantile(:,i)=Forecast+Segment;
            end
            AfterProcessQuantile=max(Quantile,MIN);
            AfterProcessQuantile=min(AfterProcessQuantile,MAX);
            
            Error=zeros(length(Forecast),99);
            for i=1:1:length(Forecast)
                for q=1:1:99
                    
                    if TestY_D2(i) < AfterProcessQuantile(i,q)
                        Error(i,q)=(1-q/100)*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    else
                        Error(i,q)=q/100*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    end
                end
            end
            QP(t,1)=mean(mean(Error));
        end
        
        [Val,Loc]=min(QP);
        Parameter=DeCoeffStep*Loc;
        
    case 'Norm'
        STDStep=0.0025;
        StepNumber=500;
        
        QP=zeros(StepNumber,1);
        parfor t=1:StepNumber
            STD=STDStep*t;
            TestY_D2=TestY;
            
            TotalError=zeros(size(Forecast,1),1);
            for i=1:1:size(Forecast,1)
                RND=Forecast(i)+STD*randn(5000,1);
                Quantile=quantile(RND,99);
                
                AfterProcessQuantile=max(Quantile,MIN);
                AfterProcessQuantile=min(AfterProcessQuantile,MAX);
                
                if Forecast(i)==0
                    AfterProcessQuantile=zeros(1,99);
                end
                
                Error=zeros(99,1);
                for q=1:1:99
                    if TestY_D2(i) < AfterProcessQuantile(q)
                        Error(q,1)=(1-q/100)*abs(TestY_D2(i)-AfterProcessQuantile(q));
                    else
                        Error(q,1)=q/100*abs(TestY_D2(i)-AfterProcessQuantile(q));
                    end
                end
                TotalError(i,1)=sum(Error);
            end
            QP(t,1)=sum(TotalError)/99/length(Forecast);
        end
        
        [Val,Loc]=min(QP);
        Parameter=STDStep*Loc;
        
        
        
    case 'Laplace'        
        
        STDStep=0.001; % 0.0005
        StepNumber=500;
        Seed=rand(5000,1)-0.5;
        
        QP=zeros(StepNumber,1);
        for t=1:StepNumber
            STD=STDStep*t;
            b=sqrt(STD/2);
            Distribution=-b*sign(Seed).*log(1-2*abs(Seed));
            TestY_D2=TestY;
            
            RND=bsxfun(@plus,Distribution,Forecast');
            
            if size(Forecast,1)==1
                Quantile=quantile(RND,99);
            else
                Quantile=quantile(RND,99)';
            end
            
            AfterProcessQuantile=max(Quantile,MIN);
            AfterProcessQuantile=min(AfterProcessQuantile,MAX);
            
            Error=zeros(size(Forecast,1),99);
            for i=1:1:size(Forecast,1)
                for q=1:1:99
                    if TestY_D2(i) < AfterProcessQuantile(i,q)
                        Error(i,q)=(1-q/100)*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    else
                        Error(i,q)=q/100*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    end
                end
            end
            QP(t,1)=mean(mean(Error));
        end
        [Val,Loc]=min(QP);
        Parameter=STDStep*Loc;
        
        
    case 'Gamma'
        
        % Left
        STDStep=0.0008;
        StepNumber=500;
        Seed=rand(10000,1)-0.5;
        
        QP=zeros(StepNumber,1);
        for t=1:StepNumber
            STD=STDStep*t;
            b=sqrt(STD/2);
            Distribution=-b*sign(Seed).*log(1-2*abs(Seed));
            TestY_D2=TestY;
            
            RND=bsxfun(@plus,Distribution,Forecast');
            Quantile=quantile(RND,99)';
            
            AfterProcessQuantile=max(Quantile,MIN);
            AfterProcessQuantile=min(AfterProcessQuantile,MAX);
        
            Error=zeros(size(Forecast,1),99);
            for i=1:1:size(Forecast,1)
                for q=1:1:49
                    if TestY_D2(i) < AfterProcessQuantile(i,q)
                        Error(i,q)=(1-q/100)*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    else
                        Error(i,q)=q/100*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    end
                end
            end
            QP(t,1)=mean(mean(Error));
        end
        [Val,Loc]=min(QP);
        Parameter(1)=STDStep*Loc;
        
        figure(2);plot(QP);
        title(Parameter(1));
        grid minor;
        
        
        
        % Right
        STDStep=0.0008;
        StepNumber=500;
        Seed=rand(5000,1)-0.5;
        
        QP=zeros(StepNumber,1);
        for t=1:StepNumber
            STD=STDStep*t;
            b=sqrt(STD/2);
            Distribution=-b*sign(Seed).*log(1-2*abs(Seed));
            TestY_D2=TestY;
            
            RND=bsxfun(@plus,Distribution,Forecast');
            Quantile=quantile(RND,99)';
            
            AfterProcessQuantile=max(Quantile,MIN);
            AfterProcessQuantile=min(AfterProcessQuantile,MAX);
        
            Error=zeros(size(Forecast,1),99);
            for i=1:1:size(Forecast,1)
                for q=50:1:99
                    if TestY_D2(i) < AfterProcessQuantile(i,q)
                        Error(i,q)=(1-q/100)*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    else
                        Error(i,q)=q/100*abs(TestY_D2(i)-AfterProcessQuantile(i,q));
                    end
                end
            end
            QP(t,1)=mean(mean(Error));
        end
        [Val,Loc]=min(QP);
        Parameter(2)=STDStep*Loc;
        
        figure(3);plot(QP);
        title(Parameter(2));
        grid minor;

        
        
        
end

if ~strcmp(QuantileMode,'Gamma')
    figure(4);plot(STDStep* [1:1:length(QP)] , QP );
    title(Parameter);
    grid on;
    
    Parameter(2)=Parameter(1);
end




