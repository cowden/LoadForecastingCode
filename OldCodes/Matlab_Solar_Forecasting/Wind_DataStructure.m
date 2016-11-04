function [FinalTrainX]=Wind_DataStructure(OriginalTrainX,PureTrainDates,Command,InputType,DaySum)

Days=length(OriginalTrainX{1,1})/24;
Hours=repmat((1:1:24)',Days,1);

K=Command.InputSmooth;

%% Structure Gen
DataStructure=cell(1,11);
DataStructure{1,11}=[Hours PureTrainDates];

if InputType==1    % Pure
    for Z=1:1:10
        DataStructure{1,Z}=[ OriginalTrainX{1,Z}(:,1) OriginalTrainX{1,Z}(:,2) OriginalTrainX{1,Z}(:,3) OriginalTrainX{1,Z}(:,4)];
    end
      
elseif InputType==2  % Min
    
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Logistic100=1./(1+exp(-(Amp100-7)));
       
        Set=[Amp10 Amp100 Logistic100];
        
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end
        
elseif InputType==3  % Min
    
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Logistic100=1./(1+exp(-(Amp100-7)));
       
        Set=[Amp10 Amp100 OriginalTrainX{1,Z}(:,1) OriginalTrainX{1,Z}(:,2) OriginalTrainX{1,Z}(:,3) OriginalTrainX{1,Z}(:,4) ...
            Logistic100 Angle10 Angle100 ];
        
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end
        
elseif InputType==4
    
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[Amp100 Amp10 ...
            sin(Angle10) cos(Angle10) sin(Angle100) cos(Angle100) ...
            Amp100./Amp10 Logistic100];
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end
        
elseif InputType==5
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[Amp100 OriginalTrainX{1,Z}(:,1) OriginalTrainX{1,Z}(:,2) OriginalTrainX{1,Z}(:,3)...
            OriginalTrainX{1,Z}(:,4) ...
            sin(Angle10) cos(Angle10) sin(Angle100) cos(Angle100) ...
            Amp10 Logistic100 Amp100./Amp10];
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end
    
elseif InputType==6
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[Amp100 Amp10   Amp10.^2 Amp10.^3 Amp100.^2 Amp100.^3 ...
            Angle10 Angle100 sin(Angle10) cos(Angle10) sin(Angle100) cos(Angle100) Amp100./Amp10 Logistic100];
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end
    
elseif InputType==7
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[Amp100 OriginalTrainX{1,Z}(:,1) OriginalTrainX{1,Z}(:,2) OriginalTrainX{1,Z}(:,3) OriginalTrainX{1,Z}(:,4) ...
            Amp10   Amp10.^2 Amp10.^3 Amp100.^2 Amp100.^3 ...
            Angle10 Angle100 sin(Angle10) cos(Angle10) sin(Angle100) cos(Angle100) ...
            Amp100./Amp10 Logistic100];
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end

    Amp100Loc=zeros(10,1);
    Amp10Loc=zeros(10,1);
    for i=1:1:10
        Amp100Loc(i)=1+size(Set,2)*(i-1);
        Amp10Loc(i)=2+size(Set,2)*(i-1);
    end
    
elseif InputType==8
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Diff100=[0 ; diff(Amp100)];                   Diff10=[0 ; diff(Amp10)];        
        
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[Amp100 OriginalTrainX{1,Z}(:,1) OriginalTrainX{1,Z}(:,2) OriginalTrainX{1,Z}(:,3) OriginalTrainX{1,Z}(:,4) ...
            (OriginalTrainX{1,Z}(:,1)).^2 (OriginalTrainX{1,Z}(:,2)).^2 ...
            (OriginalTrainX{1,Z}(:,3)).^2 (OriginalTrainX{1,Z}(:,4)).^2 ...
            Amp10   Amp10.^2 Amp10.^3 Amp100.^2 Amp100.^3 Diff100  Diff10...
            Angle10 Angle100 sin(Angle10) cos(Angle10) sin(Angle100) cos(Angle100) ...
            Amp100./Amp10 Logistic100];
        
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end

    Amp100Loc=zeros(10,1);
    Amp10Loc=zeros(10,1);
    for i=1:1:10
        Amp100Loc(i)=1+size(Set,2)*(i-1);
        Amp10Loc(i)=2+size(Set,2)*(i-1);
    end
    
    
    elseif InputType==9
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Diff100=[0 ; diff(Amp100)];                   Diff10=[0 ; diff(Amp10)];        
        
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[ OriginalTrainX{1,Z}(:,1) OriginalTrainX{1,Z}(:,2) OriginalTrainX{1,Z}(:,3) OriginalTrainX{1,Z}(:,4) ...
            (OriginalTrainX{1,Z}(:,1)).^2 (OriginalTrainX{1,Z}(:,2)).^2 ...
            (OriginalTrainX{1,Z}(:,3)).^2 (OriginalTrainX{1,Z}(:,4)).^2 ...
            (OriginalTrainX{1,Z}(:,1)).^3 (OriginalTrainX{1,Z}(:,2)).^3 ...
            (OriginalTrainX{1,Z}(:,3)).^3 (OriginalTrainX{1,Z}(:,4)).^3 ...
            Amp10 Amp10.^2 Amp10.^3 Amp100 Amp100.^2 Amp100.^3 Diff100  Diff10...
            Logistic100];
         
         % Angle10 Angle100  sin(Angle10) cos(Angle10) sin(Angle100) cos(Angle100) 
        
        
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end

    Amp100Loc=zeros(10,1);
    Amp10Loc=zeros(10,1);
    for i=1:1:10
        Amp100Loc(i)=1+size(Set,2)*(i-1);
        Amp10Loc(i)=2+size(Set,2)*(i-1);
    end
    
    
elseif InputType==10
    for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[Amp10  Amp10.^2 Amp10.^3 Amp100  Amp100.^2 Amp100.^3 Logistic100];
        
       
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end
    
    
elseif InputType==11
     for Z=1:1:10
        XValue10=OriginalTrainX{1,Z}(:,1);           XValue10=smooth(XValue10,K);
        YValue10=OriginalTrainX{1,Z}(:,2);           YValue10=smooth(YValue10,K);
        
        XValue100=OriginalTrainX{1,Z}(:,3);          XValue100=smooth(XValue100,K);
        YValue100=OriginalTrainX{1,Z}(:,4);          YValue100=smooth(YValue100,K);
        
        Amp10=sqrt(XValue10.^2+YValue10.^2);             Angle10=atan2(YValue10,XValue10);
        Amp100=sqrt(XValue100.^2+YValue100.^2);          Angle100=atan2(YValue100,XValue100);
        
        Diff100=[0 ; diff(Amp100)];                   Diff10=[0 ; diff(Amp10)];        
        
        Logistic100=1./(1+exp(-(Amp100-7)));
        
        Set=[ OriginalTrainX{1,Z}(:,1) OriginalTrainX{1,Z}(:,2) OriginalTrainX{1,Z}(:,3) OriginalTrainX{1,Z}(:,4) ...
            (OriginalTrainX{1,Z}(:,1)).^2 (OriginalTrainX{1,Z}(:,2)).^2 ...
            (OriginalTrainX{1,Z}(:,3)).^2 (OriginalTrainX{1,Z}(:,4)).^2 ...            
            (OriginalTrainX{1,Z}(:,1)).^3 (OriginalTrainX{1,Z}(:,2)).^3 ...
            (OriginalTrainX{1,Z}(:,3)).^3 (OriginalTrainX{1,Z}(:,4)).^3 ...            
            Amp10 Amp10.^2 Amp10.^3 Amp100 Amp100.^2 Amp100.^3 Diff100  Diff10...
            Logistic100 Angle10 Angle100];
         
         % Angle10 Angle100  sin(Angle10) cos(Angle10) sin(Angle100) cos(Angle100) 
                
        for j=1:1:size(Set,2)
            Set(:,j)=smooth(Set(:,j),K);
        end
        DataStructure{1,Z}=Set;
    end

    Amp100Loc=zeros(10,1);
    Amp10Loc=zeros(10,1);
    for i=1:1:10
        Amp100Loc(i)=1+size(Set,2)*(i-1);
        Amp10Loc(i)=2+size(Set,2)*(i-1);
    end
    
end

TotalTrainX=cell2mat(DataStructure);


%% Daily Combiner
if DaySum==0
    FinalTrainX=TotalTrainX;
    
else
    PreCase=cell(1,DaySum);
    for i=DaySum:-1:1
        PreCase{1,i}=[repmat(TotalTrainX(1,:),i,1); TotalTrainX(1:end-i,:)];
    end
    PostCase=cell(1,DaySum);
    for i=1:1:DaySum
        PostCase{1,i}=[TotalTrainX(i+1:end,:) ; repmat(TotalTrainX(end,:),i,1)];
    end
    Case{1,1}=TotalTrainX;
    FinalTrainX=cell2mat([ PreCase  Case PostCase]);
    
end

