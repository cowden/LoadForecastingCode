function [GroupForList]=Wind_ReGroup(Forecast,QuantileGroupSize)
%% Classification
GroupForList=cell(QuantileGroupSize,1);

%%
if QuantileGroupSize==1
    GroupForList{1,1}=(1:1:size(Forecast,1))';
    
elseif QuantileGroupSize==2
    
elseif QuantileGroupSize==3
    GroupForList{1,1}=find(Forecast < 0.2);
    GroupForList{2,1}=find(0.2<=Forecast & Forecast < 0.8);
    GroupForList{3,1}=find(0.8<=Forecast);
    
elseif QuantileGroupSize==4
    GroupForList{1,1}=find(Forecast < 0.1);
    GroupForList{2,1}=find(0.1<=Forecast & Forecast < 0.5);
    GroupForList{3,1}=find(0.5<=Forecast & Forecast < 0.8);
    GroupForList{4,1}=find(0.8<=Forecast );
    
elseif QuantileGroupSize==5
    GroupForList{1,1}=find(Forecast < 0.2);
    GroupForList{2,1}=find(0.2<=Forecast & Forecast < 0.4);
    GroupForList{3,1}=find(0.4<=Forecast & Forecast < 0.6);
    GroupForList{4,1}=find(0.6<=Forecast & Forecast < 0.8);
    GroupForList{5,1}=find(0.8<=Forecast );
    
elseif QuantileGroupSize==6
    
    
elseif QuantileGroupSize==10
    GroupForList{1,1}=find(Forecast < 0.1);
    GroupForList{2,1}=find(0.1<=Forecast & Forecast < 0.2);
    GroupForList{3,1}=find(0.2<=Forecast & Forecast < 0.3);
    GroupForList{4,1}=find(0.3<=Forecast & Forecast < 0.4);
    GroupForList{5,1}=find(0.4<=Forecast & Forecast < 0.5);
    GroupForList{6,1}=find(0.5<=Forecast & Forecast < 0.6);
    GroupForList{7,1}=find(0.6<=Forecast & Forecast < 0.7);
    GroupForList{8,1}=find(0.7<=Forecast & Forecast < 0.8);
    GroupForList{9,1}=find(0.8<=Forecast & Forecast < 0.9);
    GroupForList{10,1}=find(0.9<=Forecast );
    
elseif QuantileGroupSize==20    
    GroupForList{1,1}=find(Forecast < 0.05);
    GroupForList{2,1}=find(0.05<=Forecast & Forecast < 0.10);
    GroupForList{3,1}=find(0.10<=Forecast & Forecast < 0.15);
    GroupForList{4,1}=find(0.15<=Forecast & Forecast < 0.20);
    GroupForList{5,1}=find(0.20<=Forecast & Forecast < 0.25);
    GroupForList{6,1}=find(0.25<=Forecast & Forecast < 0.30);
    GroupForList{7,1}=find(0.30<=Forecast & Forecast < 0.35);
    GroupForList{8,1}=find(0.35<=Forecast & Forecast < 0.40);
    GroupForList{9,1}=find(0.40<=Forecast & Forecast < 0.45);
    GroupForList{10,1}=find(0.45<=Forecast & Forecast < 0.50);
    GroupForList{11,1}=find(0.50<=Forecast & Forecast < 0.55);
    GroupForList{12,1}=find(0.55<=Forecast & Forecast < 0.60);
    GroupForList{13,1}=find(0.60<=Forecast & Forecast < 0.65);
    GroupForList{14,1}=find(0.65<=Forecast & Forecast < 0.70);
    GroupForList{15,1}=find(0.70<=Forecast & Forecast < 0.75);
    GroupForList{16,1}=find(0.75<=Forecast & Forecast < 0.80);
    GroupForList{17,1}=find(0.80<=Forecast & Forecast < 0.85);
    GroupForList{18,1}=find(0.85<=Forecast & Forecast < 0.90);
    GroupForList{19,1}=find(0.90<=Forecast & Forecast < 0.95);    
    GroupForList{20,1}=find(0.95<=Forecast );
    
end

