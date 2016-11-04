function [TestOutput,TrainOutput]=Gauss(TrainX,TrainY,TestX,Parameters)

startup;


N=size(TrainY,1);

YMean=mean(TrainY);
ABS=max(abs(YMean));
NewTrainY=(TrainY-YMean)/ABS;


%% Initial Hyper
%hyp.cov = [0; 0]; 
hyp.mean = [0; zeros(size(TrainX,2),1)];
hyp.lik = log(0.1);

%% Cov Function
%covfunc=@covSEard; 
L = rand(size(TrainX,2),1);
D=size(TrainX,2);
sf = 2;
al = 2;
ell = .9; 

% cga = {@covSEard};   hypga = log([L;sf]);       % Gaussian with ARD
cn  = {'covNoise'}; sn = .1;  hypn = log(sn);  % one hyperparameter
cc  = {@covConst};   sf = 2;  hypc = log(sf); % function handles OK
% cla = {'covLINard'}; L = rand(D,1); hypla = log(L);  % linear (ARD)
%cma = {@covMaternard,3};  hypma = log([ell;sf]); % Matern class d=3
%cpc = {'covCos'}; p = 2; hypcpc = log([p;sf]);         % cosine cov
% cpe = {'covPeriodic'}; p = 2; hyppe = log([ell;p;sf]);   % periodic
cri = {@covRQiso};          hypri = log([ell;sf;al]);   % isotropic


cgi = {'covSEiso'};  hypgi = log([ell;sf]);    % isotropic Gaussian
cli = {'covLINiso'}; l = rand(1);   hypli = log(l);    % linear iso

csu = {'covSum',{cgi,cli,cn,cc,cri}}; hypsu = [hypgi ; hypli; hypn; hypc ; hypri];      % sum


hyp.cov=hypsu;
covfunc=csu;

likfunc = @likGauss;

%%
nu = fix(N/2); 
u = linspace(-1.3,1.3,nu)';
%covfuncF={@covFITC, {covfunc}, u};
meanfunc = {@meanSum, {@meanLinear, @meanConst}};
Inference=@infExact;  % @infExact;   %  infFITC


SampleRatio=Parameters.SampleRatio;
SampleN=fix(SampleRatio*size(TrainX,1));
SampleTrain=randperm(size(TrainX,1),SampleN);


hyp = minimize(hyp, @gp, -100,Inference, meanfunc, covfunc, likfunc, TrainX(SampleTrain,:), NewTrainY(SampleTrain,:));
[GP_Output,~,~,~] = gp(hyp, Inference, meanfunc, covfunc, likfunc, TrainX, NewTrainY, [TrainX;TestX]);


%%
TrainOutput=GP_Output(1:length(TrainX),:);
TrainOutput=TrainOutput*ABS+YMean;



%%
TestOutput=GP_Output(length(TrainX)+1:end,:);
TestOutput=TestOutput*ABS+YMean;





