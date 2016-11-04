function [OptimalSTD]=NewQuantileEvaluation(BestPointForecast,ZonalTrainY,QuantileMinMax,QuantileMode)

%%
Length=length(BestPointForecast);

OptimalSTD=zeros(Length,1);




switch QuantileMode
    case 'Uniform'
        
        STDStep=0.00012;
        StepNumber=100;
        
        for r=1:1:length(BestPointForecast)
            Forecast=BestPointForecast(r,1);
            Y=ZonalTrainY(r,1);
            
%             if Forecast==0
%                 OptimalSTD(r,1)=0;
%                 continue;
%             end
            
            QP=zeros(StepNumber,1);
            for t=1:StepNumber
                DeCoeff=STDStep*t;
                
                Quantile=zeros(1,99);
                for i=1:1:99
                    Segment= DeCoeff*(i-50);
                    Quantile(1,i)=Forecast+Segment;
                end
                AfterProcessQuantile=max(Quantile,QuantileMinMax(r,1));
                AfterProcessQuantile=min(AfterProcessQuantile,QuantileMinMax(r,2));
                
                Error=zeros(99,1);
                for q=1:1:99
                    if Y < AfterProcessQuantile(q)
                        Error(q,1)=(1-q/100)*abs(Y-AfterProcessQuantile(q));
                    else
                        Error(q,1)=q/100*abs(Y-AfterProcessQuantile(q));
                    end
                end
                QP(t,1)=mean(Error);
            end
            [Val,Loc]=min(QP);
            OptimalSTD(r,1)=STDStep*Loc;
        end
        
        figure(2);plot(QP);
        
        
        
    case 'Norm'
        STDStep=0.0025;
        StepNumber=500;
        
        for r=1:1:length(BestPointForecast)
            display(r);
            Forecast=BestPointForecast(r,1);
            Y=ZonalTrainY(r,1);
%             
%             if Forecast==0
%                 OptimalSTD(r,1)=0;
%                 continue;
%             end
            
            QP=zeros(StepNumber,1);
            for t=1:StepNumber
                STD=STDStep*t;
                
                RND=Forecast+STD*randn(5000,1);
                Quantile=quantile(RND,99);
                
                AfterProcessQuantile=max(Quantile,QuantileMinMax(r,1));
                AfterProcessQuantile=min(AfterProcessQuantile,QuantileMinMax(r,2));
                
                Error=zeros(99,1);
                for q=1:1:99
                    if Y < AfterProcessQuantile(q)
                        Error(q,1)=(1-q/100)*abs(Y-AfterProcessQuantile(q));
                    else
                        Error(q,1)=q/100*abs(Y-AfterProcessQuantile(q));
                    end
                end
                QP(t,1)=mean(Error);
            end
            figure(2);plot(QP);grid on;
            
            
            
            [Val,Loc]=min(QP);
            OptimalSTD(r,1)=STDStep*Loc;
            
        end
        
    case 'Laplace'
        
        STDStep=0.0005; % 0.0005
        StepNumber=200;
        Seed=rand(5000,1)-0.5;
        
        for r=1:1:length(BestPointForecast)
            display(r);
            Forecast=BestPointForecast(r,1);
            Y=ZonalTrainY(r,1);
            
            if Forecast==0
                OptimalSTD(r,1)=0;
                continue;
            end
            
            
            
            QP=zeros(StepNumber,1);
            for t=1:StepNumber
                STD=STDStep*t;
                b=sqrt(STD/2);
                Distribution=-b*sign(Seed).*log(1-2*abs(Seed));
                
                RND=bsxfun(@plus,Distribution,Forecast');
                Quantile=quantile(RND,99)';
                
                AfterProcessQuantile=max(Quantile,min(ZonalTrainY));
                AfterProcessQuantile=min(AfterProcessQuantile,max(ZonalTrainY));
                
                Error=zeros(99,1);
                for q=1:1:99
                    if Y < AfterProcessQuantile(q)
                        Error(q,1)=(1-q/100)*abs(Y-AfterProcessQuantile(q));
                    else
                        Error(q,1)=q/100*abs(Y-AfterProcessQuantile(q));
                    end
                end
                QP(t,1)=mean(Error);
            end
            [Val,Loc]=min(QP);
            OptimalSTD(r,1)=STDStep*Loc;
            
        end
        
    case 'Gamma'
        
        % Left
        STDStep=0.001;
        StepNumber=200;
        Seed=rand(5000,1)-0.5;
        
        QP=zeros(StepNumber,1);
        for t=1:StepNumber
            STD=STDStep*t;
            b=sqrt(STD/2);
            Distribution=-b*sign(Seed).*log(1-2*abs(Seed));
            TestY_D2=TestY;
            
            RND=bsxfun(@plus,Distribution,Forecast');
            Quantile=quantile(RND,99)';
            
            AfterProcessQuantile=max(Quantile,min(TestY));
            AfterProcessQuantile=min(AfterProcessQuantile,max(TestY));
            
            Error=zeros(length(Forecast),99);
            for i=1:1:length(Forecast)
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
        Parameter.Left=STDStep*Loc;
        
        figure(2);plot(QP);
        title(Parameter.Left);
        grid minor;
        
        
        
        % Right
        STDStep=0.001;
        StepNumber=200;
        Seed=rand(5000,1)-0.5;
        
        QP=zeros(StepNumber,1);
        for t=1:StepNumber
            STD=STDStep*t;
            b=sqrt(STD/2);
            Distribution=-b*sign(Seed).*log(1-2*abs(Seed));
            TestY_D2=TestY;
            
            RND=bsxfun(@plus,Distribution,Forecast');
            Quantile=quantile(RND,99)';
            
            AfterProcessQuantile=max(Quantile,min(TestY));
            AfterProcessQuantile=min(AfterProcessQuantile,max(TestY));
            
            Error=zeros(length(Forecast),99);
            for i=1:1:length(Forecast)
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
        Parameter.Right=STDStep*Loc;
        
        figure(2);plot(QP);
        title(Parameter.Left);
        grid minor;
        
        figure(3);plot(QP);
        title(Parameter.Right);
        grid minor;
        
        
        
        
end









