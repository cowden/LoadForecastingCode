function [UniqueNorm]=Normalization(Input)


NomalMode='Standard';

%% Remove Same Hours
InputSTD=std(Input);
ZeroHours=find(InputSTD==0);
Unique=setxor((1:1:size(Input,2))',ZeroHours);
UniqueInput=Input(:,Unique);



%% Mean Extraction
Normal=bsxfun(@rdivide,UniqueInput,max(abs(UniqueInput)));
Mean=mean(Normal);
ZeroMean=bsxfun(@minus,Normal,Mean);


switch NomalMode
    case 'Standard'
        
        %% Standard Deviation Extraction
        STD=std(ZeroMean);
        Output=bsxfun(@rdivide,ZeroMean,STD);
        
    case 'PCA'
        
        COV=cov(ZeroMean);
        [V,D]=eig(COV);
        Diagonal=1./sqrt(diag(D));
        New=diag(Diagonal)*V'*ZeroMean';
        Output=New';
        
end

Output=round(Output*1e5)/1e5;
UniqueNorm=unique(Output','rows')';
