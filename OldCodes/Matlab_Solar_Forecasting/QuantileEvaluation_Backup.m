function [Parameter,Val]=QuantileEvaluation(TestY,Forecast,MIN,MAX,QuantileMode)

Golden=(1+sqrt(5))/2;


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
        
        SeedNumber=randn(10000,1);
        SeedQuantile=quantile(SeedNumber,99);
        
        StepN=2000;
        STDCand=0.0010*(1:1:StepN);
        %% Initial Conditions
        x1=STDCand(1);
        x3=STDCand(end);
        a=(x3-x1)/(1+Golden);
        x2=x1+a;
        b=x3-x1-a;
        c=a^2/b;
        x4=x2+c;
        
        Saver=zeros(StepN*4,2);
        
        for t=1:1:100
            y1=Quant(x1,SeedQuantile,TestY,Forecast);
            y2=Quant(x2,SeedQuantile,TestY,Forecast);
            y3=Quant(x3,SeedQuantile,TestY,Forecast);
            y4=Quant(x4,SeedQuantile,TestY,Forecast);
            
            Saver(t*4,1)=x1;
            Saver(t*4,2)=y1;
            Saver(t*4-1,1)=x2;
            Saver(t*4-1,2)=y2;
            Saver(t*4-2,1)=x3;
            Saver(t*4-2,2)=y3;
            Saver(t*4-3,1)=x4;
            Saver(t*4-3,2)=y4;
            
            if abs(y2-y3) < 10
                break;
            end
            
            
            if y4 > y2
                %     x1=x1;
                x3=x4;
                x4=x2;
                
                
                % find x2
                a=x3-x4;
                b=x4-x1;
                c=a^2/b;
                x2=x4-c;
                
            elseif y4 < y2
                x1=x2;
                x2=x4;
                %x3=x3;
                
                
                % find x4
                a=x2-x1;
                b=x3-x2;
                c=a*b/(a+b);
                x4=x2+c;
                
            end
        end
        
        Parameter=x2;
        
        Val=y2;
        figure(4);
        scatter(Saver(:,1),Saver(:,2));
        grid on;
        %
        %
        %         STDStep=0.0025;
        %         StepNumber=500;
        %
        %         QP=zeros(StepNumber,1);
        %         parfor t=1:StepNumber
        %             STD=STDStep*t;
        %             TestY_D2=TestY;
        %
        %             TotalError=zeros(size(Forecast,1),1);
        %             for i=1:1:size(Forecast,1)
        %                 RND=Forecast(i)+STD*randn(5000,1);
        %                 AfterProcessQuantile=quantile(RND,99);
        %
        %                 Error=zeros(99,1);
        %                 for q=1:1:99
        %                     if TestY_D2(i) < AfterProcessQuantile(q)
        %                         Error(q,1)=(1-q/100)*abs(TestY_D2(i)-AfterProcessQuantile(q));
        %                     else
        %                         Error(q,1)=q/100*abs(TestY_D2(i)-AfterProcessQuantile(q));
        %                     end
        %                 end
        %                 TotalError(i,1)=sum(Error);
        %             end
        %             QP(t,1)=sum(TotalError)/99/length(Forecast);
        %         end
        %
        %         [Val,Loc]=min(QP);
        %         Parameter=STDStep*Loc;
        %
        
        
    case 'Laplace'
        

        Seed=rand(10000,1)-0.5;
        SeedLap=-sign(Seed).*log(1-2*abs(Seed));
        SeedQuantile=quantile(SeedLap,99);
       

        StepN=2000;
        STDCand=0.0001*(1:1:StepN);
        %% Initial Conditions
        x1=STDCand(1);
        x3=STDCand(end);
        a=(x3-x1)/(1+Golden);
        x2=x1+a;
        b=x3-x1-a;
        c=a^2/b;
        x4=x2+c;
        
        Saver=zeros(StepN*4,2);
        
        for t=1:1:1000
            y1=Quant(x1,SeedQuantile,TestY,Forecast);
            y2=Quant(x2,SeedQuantile,TestY,Forecast);
            y3=Quant(x3,SeedQuantile,TestY,Forecast);
            y4=Quant(x4,SeedQuantile,TestY,Forecast);
            
            Saver(t*4,1)=x1;
            Saver(t*4,2)=y1;
            Saver(t*4-1,1)=x2;
            Saver(t*4-1,2)=y2;
            Saver(t*4-2,1)=x3;
            Saver(t*4-2,2)=y3;
            Saver(t*4-3,1)=x4;
            Saver(t*4-3,2)=y4;
            
            if abs(y2-y3) < 5
                break;
            end
            
            if y4 > y2
                %     x1=x1;
                x3=x4;
                x4=x2;
                
                % find x2
                a=x3-x4;
                b=x4-x1;
                c=a^2/b;
                x2=x4-c;
                
            elseif y4 < y2
                x1=x2;
                x2=x4;
                %x3=x3;
                
                
                % find x4
                a=x2-x1;
                b=x3-x2;
                c=a*b/(a+b);
                x4=x2+c;
                
            end
        end
        
        Parameter=x2; %max(x2,x3);
        
        Val=y2;
        figure(4);
        scatter(Saver(:,1),Saver(:,2));
        grid on;
        
        
%                 STDStep=0.0005; % 0.0005
%         StepNumber=500;
%         
        %         QP=zeros(StepNumber,1);
        %         for t=1:StepNumber
        %             STD=STDStep*t;
        %             b=sqrt(STD/2);
        %             Cloth=b*SeedQuantile;
        %             Quantile=bsxfun(@plus,Forecast,Cloth);
        %
        %             Error=zeros(size(Forecast,1),99);
        %             for i=1:1:size(Forecast,1)
        %                 for q=1:1:99
        %                     if TestY(i) < Quantile(i,q)
        %                         Error(i,q)=(1-q/100)*abs(TestY(i)-Quantile(i,q));
        %                     else
        %                         Error(i,q)=q/100*abs(TestY(i)-Quantile(i,q));
        %                     end
        %                 end
        %             end
        %             QP(t,1)=sum(sum(Error));
        %         end
        %         [Val,Loc]=min(QP);
        %         Parameter=STDStep*Loc;
        
        
    case 'Gamma'
        
        Seed=rand(10000,1)-0.5;
        SeedLap=-sign(Seed).*log(1-2*abs(Seed));
        SeedQuantile=quantile(SeedLap,99);
       
        
        
        
        
        % Left
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
            AfterProcessQuantile=quantile(RND,99)';
            
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
            AfterProcessQuantile=quantile(RND,99)';
            
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

%
% if ~strcmp(QuantileMode,'Gamma')
%     figure(4);plot(STDStep* [1:1:length(QP)] , QP );
%     title(Parameter);
%     grid on;
%
%     Parameter(2)=Parameter(1);
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [QP]=Quant(STD,SeedQuantile,TestY,Forecast)
b=sqrt(STD/2);
Cloth=b*SeedQuantile;
Quantile=bsxfun(@plus,Forecast,Cloth);
Error=zeros(size(Forecast,1),99);
for i=1:1:size(Forecast,1)
    for q=1:1:99
        if TestY(i) < Quantile(i,q)
            Error(i,q)=(1-q/100)*abs(TestY(i)-Quantile(i,q));
        else
            Error(i,q)=q/100*abs(TestY(i)-Quantile(i,q));
        end
    end
end
QP=sum(sum(Error));




% QuantSeed=(0.01:0.01:0.99);
% BaseQuantile=zeros(1,99);
% for k=1:1:99
%     if QuantSeed(k) < 0.5
%         BaseQuantile(k)= STD*log(2*QuantSeed(k));
%     else
%         BaseQuantile(k)= -STD*log(2-2*QuantSeed(k));
%     end
% end
% QuantileMatrix=bsxfun(@plus,Forecast,BaseQuantile);
% QP=QuantileErrorCal(QuantileMatrix,Forecast,TestY);



function [QP]=QuantileErrorCal(QuantileMatrix,Forecast,TestY)

TotalError=zeros(size(Forecast,1),99);
for i=1:1:size(Forecast,1)
    for q=1:1:99
        if TestY(i) < QuantileMatrix(i,q)
            TotalError(i,q)=(1-q/100)*abs(TestY(i)-QuantileMatrix(i,q));
        else
            TotalError(i,q)=q/100*abs(TestY(i)-QuantileMatrix(i,q));
        end
    end
end
QP=sum(sum(TotalError));





