function [SubL,TrainX,TrainY,TestX,TestY]=CVSplitter(TotalTrainX,ZonalTrainY,TotalList,Segment,t)



StartPoint=(t-1)*Segment+1;
EndPoint=min(t*Segment,size(ZonalTrainY,1));
SubTestList=(StartPoint:1:EndPoint)';
SubTrainList=setxor(TotalList,SubTestList);
TrainY=ZonalTrainY(SubTrainList,:);
TestY=ZonalTrainY(SubTestList,:);
TrainX=TotalTrainX(SubTrainList,:);
TestX=TotalTrainX(SubTestList,:);

SubL.TestX=length(TestX);
SubL.TrainX=length(TrainX);