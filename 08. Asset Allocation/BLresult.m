load('new.mat');
capweights=zeros(1,8);
for i=1:8
    capweights(1,i)=marketcap(1,i)/sum(marketcap);
end;

rets=zeros(59,8);
for i=1:8
    rets(:,i)=prices(2:end,i)./prices(1:end-1,i)-1;
end

rf=0.0045; %risk free rate 
mu=mean(rets);
sigma=cov(rets-rf);

expected_ret=mu*capweights';
lambda=(expected_ret-rf)/(capweights*sigma*capweights');
pi=lambda*sigma*capweights';
w_pi=inv(lambda*sigma)*pi;

totalcap1=marketcap(1,3)+marketcap(1,5);
weightedpi_1=(marketcap(1,3)/totalcap1)*pi(3,1)+(marketcap(1,5)/totalcap1)*pi(5,1);

totalcap2=marketcap(1,7)+marketcap(1,8);
weightedpi_2=(marketcap(1,7)/totalcap2)*pi(7,1)+(marketcap(1,8)/totalcap2)*pi(8,1);

weighted_difference=weightedpi_2-weightedpi_1;

Q=[0.003;0.001;0.0015];
p=[0,0,0,0,0,1,0,0;-1,1,0,0,0,0,0,0;0,0,-0.9961,0,-0.0039,0,0.2872,0.7128];
tau=0.025;

omega=zeros(3,3);
w1=p(1,:)*sigma*p(1,:)'*tau;
w2=p(2,:)*sigma*p(2,:)'*tau;
w3=p(3,:)*sigma*p(3,:)'*tau;
omega(1,1)=w1;
omega(2,2)=w2;
omega(3,3)=w3;

first=inv(inv(tau*sigma)+p'*inv(omega)*p);
second=[inv(tau*sigma)*pi+p'*inv(omega)*Q];
newreturn=first*second;

newweight=inv(lambda*sigma)*newreturn;%new recommend weight of portofolio

pig=figure;
plot(capweights,'g')
hold on 
plot(newweight,'b')
legend('initial capweights','newweight')
