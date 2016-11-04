clc

% 0.0403

%% Program Mode
QuantileMode='Gamma';   % Same
OperationMode='0';    %2: Coeff finding % 1: Test 0: Actual Submission
EnsemMode=1;                  % Weig hted
QuantileGroupSize=10;
Continuous=0;
NumSeg=20;
TestTime=5;
ZoneList=[7 1 9 8 2 3 6 4 5 10];


%% System Parameters
Task=14;                     % Number of Weeks
TargetMonth=Task-3;              % Target Month we will forecast

%% Solution Loading
SolFile=strcat('Sol',num2str(Task));
load(SolFile,'-mat');

%% Data Reader
[PureTrainX,OriginalTrainX,OriginalTrainY,TotalDates,PureTrainDates]=Wind_DataReader(Task,TargetMonth);
LTrainX=length(PureTrainX{1,1});
LTestX=length(OriginalTrainX{1,1})-LTrainX;

%% Zonal Output Smooth Factor
ZonalSmoothFactor=[3 3 5 5 5 5 5 5 5 5];

%% Segmentation
Segment=ceil(size(OriginalTrainY{1,1},1)/NumSeg);
FinalForecastSeven=[];
MiddleForecastSeven=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Point Forecasting
if strcmp(OperationMode,'0')     % Everything with a Weighted averaging
    
    CVForecastSaver=zeros(TestTime * Segment,10);
    CVSolutionSaver=zeros(TestTime * Segment,10);
    FinalForecastSaver=zeros(LTestX,10);
    TestXSaver=cell(10,1);
    TrainXSaver=cell(10,1);
    
    ZonalQuantile=cell(10,1);
    ZonalRMSEMain=cell(10,1);
    
    for Z=ZoneList
        OutputSmooth=ZonalSmoothFactor(Z);
        ZonalTrainY=OriginalTrainY{1,Z};    % Zonal means a single zone value
        RunSet=randperm(NumSeg,TestTime);
        TotalList=(1:1:size(ZonalTrainY,1))';
                
        RawPrediction=cell(100,TestTime);
        RawSolution=cell(100,TestTime);
        ModelRMSE=zeros(100,1);
        c=0;
        
        for index=1
            [Command]=Wind_CommandCenter(index,Z);
            %Command.LTrainX=LTrainX;
            %Command.LTestX=LTestX;
            Command.OutputSmooth=OutputSmooth;
            
            for InputType=Command.InputType
                for DaySum=Command.DaySum
                    [TotalTrainXOrg]=Wind_DataStructure(PureTrainX,PureTrainDates,Command,InputType,DaySum);
                    
                    [TotalTrainX1]=Normalization([TotalTrainXOrg MiddleForecastSeven ]);
                                        
                    for FunName=Command.FunGroup
                        
                        c=c+1;
                        
                        AccumError=cell(1,TestTime);
                        for t=1:1:TestTime
                            tt=RunSet(t);
                            tic;
                            [SubL,TrainXOrgn,TrainYOrgn,TestX,TestY]=CVSplitter(TotalTrainX1,ZonalTrainY,TotalList,Segment,tt);
                            
                            % OutLier Ditection
                            [FeelSoGood]=OutlierDitection(TrainXOrgn,TrainYOrgn,'Ridge',Command);
                            TrainX=TrainXOrgn(FeelSoGood,:);
                            TrainY=TrainYOrgn(FeelSoGood);
                            
                            % Forecast
                            [Raw,TrainError]=Wind_Forecaster(TrainX,TrainY,TestX,ZonalTrainY,FunName,Command);
                            
                            % Post Processing
                            Forecast=smooth(Raw,Command.OutputSmooth);
                            Forecast=max(Forecast,min(ZonalTrainY));
                            Forecast=min(Forecast,max(ZonalTrainY));
                            
                            AccumError{1,t}=(Forecast-TestY).^2;
                            TestRMSE=sqrt(mean((Forecast-TestY).^2));
                            
                            % Extra Information
                            NVar=size(TrainX,2);
                            ElapsedTime=toc;
                            ElapsedTime=round(ElapsedTime);
                            
                            fprintf('Zone:%d/Order:%d/Input:%d/DaySun:%d/NVar:%d/%s/%1.0fs/CV:%d/Test:%f\n',...
                                Z,index,InputType,DaySum,NVar,FunName{1},ElapsedTime,tt,TestRMSE);
                            
                            figure(1);plot(TestY);hold on;plot(Forecast,'color','r');hold off;
                            
                            % Saving
                            RawPrediction{c,t}=Raw;
                            RawSolution{c,t}=TestY;
                        end
                        ModelRMSE(c,1)=sqrt(mean(mean(cell2mat(AccumError))));
                    end
                end
            end
        end
        NUnit=c;
        RawPrediction=RawPrediction(1:NUnit,:);
        RawSolution=RawSolution(1,:);

        
        %% Ensemble Calculation
        ModelRMSE=ModelRMSE(1:NUnit,:);
        display(ModelRMSE);        
        
        if EnsemMode==1
            WeightTemp=1./(ModelRMSE);
            Weight=WeightTemp/sum(WeightTemp);
        elseif  EnsemMode==0
            Weight=ones(NUnit,1)*1/NUnit;
        end
        
        [~,BestModel]=min(ModelRMSE);
        ZonalRMSEMain{Z,1}=ModelRMSE;
        
        %% Cross Validation
        WeightTable=cell(TestTime,1);
        for t=1:1:TestTime            
            RawSeg=cell2mat(RawPrediction(:,t)');
            WeightPrediction=sum(bsxfun(@times,RawSeg,Weight'),2);
            
            % Post-processing
            AfterProcess=smooth(WeightPrediction,Command.OutputSmooth);
            AfterProcess=max(AfterProcess,min(ZonalTrainY));
            FinalBestForecast=min(AfterProcess,max(ZonalTrainY));
            
            WeightTable{t,1}=FinalBestForecast;
        end
        CVForecast=cell2mat(WeightTable(:,1));
        CVForecastSaver(:,Z)=CVForecast;
        CVSolutionSaver(:,Z)=cell2mat(RawSolution');
        %}
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Main Gain Start
        % NUnit=1;BestModel=1;            Weight=ones(NUnit,1)*1/NUnit;
        RawPrediction=zeros(LTestX,NUnit);
        c=0;
        for index=1
            [Command]=Wind_CommandCenter(index,Z);           % Decide simulation parameters
            Command.OutputSmooth=OutputSmooth;
            
            for InputType=Command.InputType
                for DaySum=Command.DaySum
                    [TotalTrainX]=Wind_DataStructure(OriginalTrainX,TotalDates,Command,InputType,DaySum);
                    [TotalTrainXTemp]=Normalization([TotalTrainX FinalForecastSeven]);
                    
                    
                    
                    ActualTrainX=TotalTrainXTemp(1:LTrainX,:);
                    TestX=TotalTrainXTemp(LTrainX+1:end,:);
                    
                    % OutLier Ditection
                    [FeelSoGood]=OutlierDitection(ActualTrainX,ZonalTrainY,'Ridge',Command);
                    TrainX=ActualTrainX(FeelSoGood,:);
                    TrainY=ZonalTrainY(FeelSoGood);
                    
                    for FunName=Command.FunGroup                        
                        
                        % Saving
                        if c==(BestModel-1)
                            TrainXSaver{Z,1}=TotalTrainX(1:LTrainX,:);
                            TestXSaver{Z,1}=TotalTrainX(   LTrainX+1:end,:);
                        end                        
                        tic;
                        c=c+1;                                            
                        
                        [Raw,TrainError]=Wind_Forecaster(TrainX,TrainY,TestX,ZonalTrainY,FunName,Command);
                        RawPrediction(:,c)=Raw;
                        
                        NVar=size(TrainX,2);
                        ElapsedTime=toc;
                        ElapsedTime=round(ElapsedTime);
                        
                        fprintf('Zone:%d/Order:%d/Input:%d/DaySum:%d/NVar:%d/%s/%1.0fs/Test:%f\n',...
                            Z,index,InputType,DaySum,NVar(1,1),FunName{1},ElapsedTime,TrainError);
                    end
                end
            end
        end
        WeightPrediction=sum(bsxfun(@times,RawPrediction,Weight'),2);
        AfterProcess=smooth(WeightPrediction,Command.OutputSmooth);
        AfterProcess=max(AfterProcess,min(ZonalTrainY));
        FinalForecast=min(AfterProcess,max(ZonalTrainY));
        FinalForecastSaver(:,Z)=FinalForecast;
               
        MiddleForecastSeven=[MiddleForecastSeven OriginalTrainY{1,Z}];
        FinalForecastSeven=[FinalForecastSeven [OriginalTrainY{1,Z};FinalForecast]];  % Expandable
            
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Solution comparison
    ZonalRMSE=zeros(10,1);
    for Z=ZoneList
        Answer=Sol(:,Z);
        ZonalRMSE(Z,1)=sqrt(mean((FinalForecastSaver(:,Z)-Answer).^2));
    end
    display(ZonalRMSE);
    
    
    save Info CVForecastSaver CVSolutionSaver FinalForecastSaver 
end

% TestXSaver TrainXSaver











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(OperationMode,'1')     % Quantile Estimator
    load Info
        
    for Z=ZoneList
        display(Z);        
        ZonalTrainY=OriginalTrainY{1,Z};    % Zonal means a single zone value
        CVForecast=CVForecastSaver(:,Z);
        CVSolution=CVSolutionSaver(:,Z);
        FinalForecast=FinalForecastSaver(:,Z);
        %TotalTrainXOrg=TrainXSaver{Z,1};
        %TestXTemp=TestXSaver{Z,1};
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Quantile Generation
        [GroupForFinalFore]=Wind_ReGroup(FinalForecast,QuantileGroupSize);
        
        % MinMax Generation
        [QuantileMinMax,MinMaxMatrix]=QuantileMinMaxGen(CVForecast,ZonalTrainY,QuantileGroupSize);
        
        if Continuous==1
            [GroupList]=Wind_ReGroup(CVForecast,QuantileGroupSize);

            Quantile=zeros(length(FinalForecast),99);
            for G=1:1:QuantileGroupSize
                SubCVForecast=CVForecast(GroupList{G,1});
                SubTestY=CVSolution(GroupList{G,1});
                E=SubTestY-SubCVForecast;
                SeedQuantile=quantile(E,99);
                                
                SelectedFinal=FinalForecast(GroupForFinalFore{G,1});
                
                SelectedQuantile=bsxfun(@plus , SelectedFinal , SeedQuantile);
                Quantile(GroupForFinalFore{G,1},:)=SelectedQuantile;

                
            end                
            
        elseif  Continuous==0
            [Quantile]=UnifiedQuantile(CVForecast,CVSolution,FinalForecast,QuantileMinMax,QuantileGroupSize,GroupForFinalFore,QuantileMode);
        end        
        ZonalQuantile{Z,1}=Quantile;
        
        
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Solution comparison
    ZonalRMSE=zeros(10,1);
    ZonalPerm=zeros(10,1);
    for Z=ZoneList
        Answer=Sol(:,Z);
        ZonalRMSE(Z,1)=sqrt(mean((FinalForecastSaver(:,Z)-Answer).^2));
        
        Quantile=ZonalQuantile{Z,1};
        QuantileError=zeros(length(Answer),99);
        for i=1:1:length(Answer)
            for q=1:1:99
                if Answer(i) < Quantile(i,q)
                    QuantileError(i,q)=(1-q/100)*abs(Answer(i)-Quantile(i,q));
                else
                    QuantileError(i,q)=q/100*abs(Answer(i)-Quantile(i,q));
                end
            end
        end
        ZonalPerm(Z)=mean(mean(QuantileError));
    end
    display(ZonalRMSE);
    display(ZonalPerm);
    
end


















