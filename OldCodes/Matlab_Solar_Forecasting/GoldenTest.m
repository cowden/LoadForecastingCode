clc
clear all
%close all


%% Parameter
Golden=(1+sqrt(5))/2;

%% Test Data
x=-5:0.01:10;
y=TestFunction(x);
%figure(1);plot(x,y);

%% Initial Conditions
x1=x(1);
x3=x(end);
a=(x3-x1)/(1+Golden);
x2=x1+a;
b=x3-x1-a;
c=a^2/b;
x4=x2+c;


%% Solve
for index=1:1:100
    display(index)
    
    y1=TestFunction(x1);
    y2=TestFunction(x2);
    y3=TestFunction(x3);
    y4=TestFunction(x4);
    
    
    if abs(y2-y3) < 0.001
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








