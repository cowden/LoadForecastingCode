function [OnlyTrainX,OriginalTrainX,OriginalTrainY,TotalDates,PureTrainDates]=Wind_DataReader(Task,TargetMonth)



%% Location Setting
if ispc
    %     DataPaths=strcat('C:\Users\hello_000\Dropbox\Forecasting Competition\Wind\Wind',num2str(Task),'\Data');
    %     addpath(DataPaths);
    %     addpath('C:\Users\hello_000\Dropbox\ToolBoxes\sqb-0.1\build');
    %     addpath('C:\Users\hello_000\Dropbox\ToolBoxes\gpml-matlab-v3.5-2014-12-08');
    %     addpath('C:\Users\hello_000\Dropbox\ToolBoxes\libsvm-3.18\windows');
    %     addpath('C:\Users\\hello_000\Dropbox\ToolBoxes\RF_MexStandalone-v0.02-precompiled\randomforest-matlab\RF_Reg_C');
        
    
    DataPaths=strcat('C:\Users\Duehee\Dropbox\Forecasting Competition\Wind\Data');
    addpath(DataPaths);
    addpath('C:\Users\Duehee\Dropbox\ToolBoxes\sqb-0.1\build');
    addpath('C:\Users\Duehee\Dropbox\ToolBoxes\gpml-matlab-v3.5-2014-12-08');    
    addpath('C:\Users\Duehee\Dropbox\ToolBoxes\libsvm-3.18\matlab');
    addpath('C:\Users\\Duehee\Dropbox\ToolBoxes\RF_MexStandalone-v0.02-precompiled\randomforest-matlab\RF_Reg_C');
    
else
    
    
    DataPaths=strcat('/Users/Keanu/Dropbox/Forecasting Competition/Wind/Data');
    addpath(DataPaths);
    addpath('/Users/Keanu/Dropbox/ToolBoxes/gpml-matlab-v3.5-2014-12-08');
    addpath('/Users/Keanu/Dropbox/ToolBoxes/sqb-0.1/build');
    addpath('/Users/Keanu/Dropbox/ToolBoxes/libsvm-3.18/matlab');
    addpath('/Users/Keanu/Dropbox/ToolBoxes/RF_MexStandalone-v0.02-precompiled/randomforest-matlab/RF_Reg_C');
    
    %
    %         DataPaths=strcat('/Users/Keanu/Documents/Dropbox/Forecasting Competition/Wind/Data');
    %         addpath(DataPaths);
    %         addpath('/Users/Keanu/Documents/Dropbox/ToolBoxes/gpml-matlab-v3.5-2014-12-08');
    %         addpath('/Users/Keanu/Documents/Dropbox/ToolBoxes/sqb-0.1/build');
    %         addpath('/Users/Keanu/Documents/Dropbox/ToolBoxes/libsvm-3.18/matlab');
    %         addpath('/Users/Keanu/Documents/Dropbox/ToolBoxes/RF_MexStandalone-v0.02-precompiled/randomforest-matlab/RF_Reg_C');
    %
end

OriginalTrainY=cell(1,10);
OriginalTrainX=cell(1,10);
OnlyTrainX=cell(1,10);
for Z=1:1:10
    TrainFileName=strcat('Task',num2str(Task),'_W_Zone',num2str(Z),'.csv');
    TrainRawData=csvread(TrainFileName,1,2);
    TempTrainX=TrainRawData(:,2:end);
    
    TestFileName=strcat('TaskExpVars',num2str(Task),'_W_Zone',num2str(Z),'.csv');
    TempTestX=csvread(TestFileName,1,2);
        
    OnlyTrainX{1,Z}=TempTrainX;
    OriginalTrainX{1,Z}=[TempTrainX; TempTestX];
    
    OriginalTrainY{1,Z}=TrainRawData(:,1);
end




%%
OneYearDays=sum(eomday(2012, (1:1:12)));
TwoYearDays=sum(eomday(2013, (1:1:TargetMonth-1)));
YearDays=[(1:1:OneYearDays) (1:1:TwoYearDays)];
Dates=repmat(YearDays,24,1);
TrainDays=reshape(Dates,[],1);


EOM=[eomday(2012, (1:1:12)) eomday(2013, (1:1:TargetMonth))];
Months=cell(12,1);
for m=1:1:length(EOM)
    
    if m>12
        m2=m-12;
    else
        m2=m;
    end
    
    Months{m,1}=m2*ones(EOM(m)*24,1);
end
Months=cell2mat(Months);

TrainMonths=Months(1:size(TempTrainX,1));
TestMonths=Months(size(TempTrainX,1)+1:end);




YearDays=(TrainDays(end)+1:1:TrainDays(end)+eomday(2013,TargetMonth));
Dates=repmat(YearDays,24,1);
TestDates=reshape(Dates,[],1);
TotalDates=[TrainDays TrainMonths; TestDates TestMonths];
PureTrainDates=[TrainDays TrainMonths];




