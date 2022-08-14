function [X0,varargout]=magnusSeparableSSC(A,a,B,b,X0,dW,T,dT,tol)
%%MAGNUSSEPARABLESSC computes Ito-stochastic Magnus expansion for given matrix
% processes A_t=A.*a(t,Z_t), B.*b(t,Z_t), such that dX_t = B_t X_t dt + A_t X_t dW_t, X_0=X0.
% If X0 is a (d x 1) vector, expmv is used instead of expm for performance
% boost. This is solved with step size control with given tolerance tol and
% maximal step size dT.
%
% Assumptions:
%   - Time grid is homogeneous.
%   - N is the number of time steps for Magnus logarithm and evaluation is
%     only at the terminal time
%
% Input:
%   A (d x d x 1 x 1 array): 
%       constant matrix 
%   B (d x d x 1 x 1 array): 
%       constant matrix
%   a (1 x 1 x N x M array): 
%       1 dimensional stochastic process
%   b (1 x 1 x N x M array): 
%       1 dimensional stochastic process
%   X0 (empty, d x 1 x 1 x (M), d x d x 1 x (M) array): 
%       initial value, if X0=[] it is assumed to be the identity matrix and
%       last axis is optional for random initial datum
%   dW (1 x 1 x N x M): 
%       increments of the Brownian motion
%   T (1 x 1 double): 
%       finite time horizon
%   dT (1 x 1 double):
%       maximal step size
%   tol (1 x 1 double): 
%       tolerance of deviation for order 2 and 3 expansion
%
% Output:
%   X (d x d x 1 x M array):
%       contains the solution at t=T
%
% See also: magnusSeparable
d=size(A,1);
N=size(dW,3)+1;
M=size(dW,4);
Mcut=ceil(M/10);

if size(X0,2)==1
    d2=floor(sqrt(d));
else
    d2=d;
end

stepSizes=[];
t=linspace(0,T,N);
dt=T/(N-1);

currTi=1;
oldTi=1;
dTind=sum(t<=dT,'all')-1;

[spExp,deviceExp,deviceLog]=compMode(size(X0,1),size(X0,2),...
                                     issparse(A) && issparse(B));
compCase=sprintf('%d%d',spExp,deviceExp);

A2=A*A;
BA=comm(B,A);
BA2=comm(B,A2);
BAA=comm(BA,A);
BAB=comm(BA,B);

if deviceLog
    A=gpuArray(A);
    B=gpuArray(B);
    A2=gpuArray(A2);
    BA=gpuArray(BA);
    BA2=gpuArray(BA2);
    BAA=gpuArray(BAA);
    BAB=gpuArray(BAB);
end

while(currTi<N)
    currDt=min(dTind,N-currTi);
    n=0;
    currTol=1e9;
    while(currTol>tol && n<10)
        currTi=oldTi+currDt;
        ind=oldTi:currTi;
        if size(X0,4)>1
            [X2,X3]=magnus23(a(1,1,ind,1:Mcut),b(1,1,ind,1:Mcut),X0(:,:,1,1:Mcut),dW(1,1,ind(1:end-1),1:Mcut),dt,t(currTi)-t(oldTi));
        else
            [X2,X3]=magnus23(a(1,1,ind,1:Mcut),b(1,1,ind,1:Mcut),X0,dW(1,1,ind(1:end-1),1:Mcut),dt,t(currTi)-t(oldTi));
        end
        currTol=isClose(X2,X3);
        if currTol>=1e9
            dTind=floor(dTind/2);
        end
%         step=floor(log10(currTol/tol));
        tempDt=floor(currDt/2);
%         tempDt2=floor(currDt/(2^(1+step)));
        if currTol>tol && (tempDt>=currDt || tempDt<=2)
            warning('Cannot reduce step size further due to too coarse Lebesgue discretization. Bad approximation possible.')
            break;
        end
%         if tempDt2>2 && tempDt2<tempDt
%             currDt=tempDt;
%         else
%             currDt=tempDt2;
%         end
        currDt=tempDt;
        n=n+1;
    end
    if Mcut<M
        if size(X0,4)>1
            X3=cat(4,X3,magnus3(a(1,1,ind,Mcut+1:end),b(1,1,ind,Mcut+1:end),X0(:,:,1,Mcut+1:end),dW(1,1,ind(1:end-1),Mcut+1:end),dt,t(currTi)-t(oldTi)));
        else
            X3=cat(4,X3,magnus3(a(1,1,ind,Mcut+1:end),b(1,1,ind,Mcut+1:end),X0,dW(1,1,ind(1:end-1),Mcut+1:end),dt,t(currTi)-t(oldTi)));
        end
    end
    stepSizes(end+1)=currTi-oldTi;
    oldTi=currTi;
    if n>10
        warning('Number of step size reduction exceeded. Bad approximation possible.')
    end
    X0=X3;
end
stepSizes=stepSizes.*dt;
if nargout>1
    varargout{1}=stepSizes;
end

function err=isClose(X2,X3)
    X2=reshape(X2,d2,d2,1,[]);
    X3=reshape(X3,d2,d2,1,[]);
    err=mean(vecnorm(vecnorm(X3(:,:,1,:)-X2(:,:,1,:),2,1),2,2)./...
        vecnorm(vecnorm(X3(:,:,1,:),2,1),2,2),4);
%     err=mean(vecnorm(vecnorm(X3(region,region,1,:)-X2(region,region,1,:),2,1),2,2),4);
    if any(isnan(err),'all')
        err=1e9;
    end
end

function [X2,X3]=magnus23(a,b,X0,dW,dt,T)
    currM=size(dW,4);
    if size(X0,2)~=1
        X2=zeros(size(A,1),size(A,1),1,currM);
        X3=zeros(size(A,1),size(A,1),1,currM);
    else
        X2=zeros(size(A,1),1,1,currM);
        X3=zeros(size(A,1),1,1,currM);
    end
    Ibdt=cumlebesgueInt(b,dt,T);
    IadW=cumstochInt(a,dW);
    Ia2dt=cumlebesgueInt(a.^2,dt,T);
    IbIadWdt=cumlebesgueInt(b.*IadW,dt,T);
    IaIbdtdW=cumstochInt(a.*Ibdt,dW);
    Ia2Ibdtdt=lebesgueInt(a.^2.*Ibdt,dt,T);
    IbIa2dtdt=lebesgueInt(b.*Ia2dt,dt,T);
    IIadWIbdtadW=stochInt(IadW.*Ibdt.*a,dW);
    IbIadW2dW=stochInt(b.*(IadW).^2,dW);
    IaIaIbdtdWdW=stochInt(a.*IaIbdtdW,dW);
    IaIbIadWdtdt=lebesgueInt(a.*IbIadWdt,dt,T);
    IbIbdtIadWdt=lebesgueInt(b.*Ibdt.*IadW,dt,T);
    IbIbIadWdtdt=lebesgueInt(b.*IbIadWdt,dt,T);
    IaIbdtdt=cumlebesgueInt(a.*Ibdt,dt,T);
    IbIaIbdtdtdt=lebesgueInt(b.*IaIbdtdt,dt,T);
    IaIbdt2dW=stochInt(a.*(Ibdt).^2,dW);

    parfor i=1:currM
        Y2=secondorder(B,A,A2,BA,Ibdt(1,1,end,i),IadW(1,1,end,i),Ia2dt(1,1,end,i),IbIadWdt(1,1,end,i),IaIbdtdW(1,1,end,i));
        Y3=thirdorder(B,A,A2,BA,BA2,BAB,BAA,...
                      Ibdt(1,1,end,i),IadW(1,1,end,i),Ia2dt(1,1,end,i),IbIadWdt(1,1,end,i),IaIbdtdW(1,1,end,i),...
                      Ia2Ibdtdt(1,1,end,i),IbIa2dtdt(1,1,end,i),IIadWIbdtadW(1,1,end,i),IbIadW2dW(1,1,end,i),IaIaIbdtdWdW(1,1,end,i),...
                      IaIbIadWdtdt(1,1,end,i),IbIbdtIadWdt(1,1,end,i),IbIbIadWdtdt(1,1,end,i),IbIaIbdtdtdt(1,1,end,i),...
                      IaIbdt2dW(1,1,end,i));
        if size(X0,4)>1
            switch compCase %spExp deviceExp
                case '00'
                    X2(:,:,1,i)=Exp(full(gather(Y2)),X0(:,:,1,i));
                    X3(:,:,1,i)=Exp(full(gather(Y3)),X0(:,:,1,i));
                case '10'
                    X2(:,:,1,i)=Exp(gather(Y2),X0(:,:,1,i));
                    X3(:,:,1,i)=Exp(gather(Y3),X0(:,:,1,i));
                case '01'
                    X2(:,:,1,i)=Exp(single(full(Y2)),gpuArray(single(X0(:,:,1,i))));
                    X3(:,:,1,i)=Exp(single(full(Y3)),gpuArray(single(X0(:,:,1,i))));
                case '11'
                    X2(:,:,1,i)=Exp(Y2,gpuArray(X0(:,:,1,i)));
                    X3(:,:,1,i)=Exp(Y3,gpuArray(X0(:,:,1,i)));
            end
        else
            switch compCase %spExp deviceExp
                case '00'
                    X2(:,:,1,i)=Exp(full(gather(Y2)),X0);
                    X3(:,:,1,i)=Exp(full(gather(Y3)),X0);
                case '10'
                    X2(:,:,1,i)=Exp(gather(Y2),X0);
                    X3(:,:,1,i)=Exp(gather(Y3),X0);
                case '01'
                    X2(:,:,1,i)=Exp(single(full(Y2)),gpuArray(single(X0)));
                    X3(:,:,1,i)=Exp(single(full(Y3)),gpuArray(single(X0)));
                case '11'
                    X2(:,:,1,i)=Exp(Y2,gpuArray(X0));
                    X3(:,:,1,i)=Exp(Y3,gpuArray(X0));
            end
        end
    end
end
function X3=magnus3(a,b,X0,dW,dt,T)
    currM=size(dW,4);
    if size(X0,2)~=1
        X3=zeros(size(A,1),size(A,1),1,currM);
    else
        X3=zeros(size(A,1),1,1,currM);
    end
    Ibdt=cumlebesgueInt(b,dt,T);
    IadW=cumstochInt(a,dW);
    Ia2dt=cumlebesgueInt(a.^2,dt,T);
    IbIadWdt=cumlebesgueInt(b.*IadW,dt,T);
    IaIbdtdW=cumstochInt(a.*Ibdt,dW);
    Ia2Ibdtdt=lebesgueInt(a.^2.*Ibdt,dt,T);
    IbIa2dtdt=lebesgueInt(b.*Ia2dt,dt,T);
    IIadWIbdtadW=stochInt(IadW.*Ibdt.*a,dW);
    IbIadW2dW=stochInt(b.*(IadW).^2,dW);
    IaIaIbdtdWdW=stochInt(a.*IaIbdtdW,dW);
    IaIbIadWdtdt=lebesgueInt(a.*IbIadWdt,dt,T);
    IbIbdtIadWdt=lebesgueInt(b.*Ibdt.*IadW,dt,T);
    IbIbIadWdtdt=lebesgueInt(b.*IbIadWdt,dt,T);
    IaIbdtdt=cumlebesgueInt(a.*Ibdt,dt,T);
    IbIaIbdtdtdt=lebesgueInt(b.*IaIbdtdt,dt,T);
    IaIbdt2dW=stochInt(a.*(Ibdt).^2,dW);

    parfor i=1:currM
        Y=thirdorder(B,A,A2,BA,BA2,BAB,BAA,...
                      Ibdt(1,1,end,i),IadW(1,1,end,i),Ia2dt(1,1,end,i),IbIadWdt(1,1,end,i),IaIbdtdW(1,1,end,i),...
                      Ia2Ibdtdt(1,1,end,i),IbIa2dtdt(1,1,end,i),IIadWIbdtadW(1,1,end,i),IbIadW2dW(1,1,end,i),IaIaIbdtdWdW(1,1,end,i),...
                      IaIbIadWdtdt(1,1,end,i),IbIbdtIadWdt(1,1,end,i),IbIbIadWdtdt(1,1,end,i),IbIaIbdtdtdt(1,1,end,i),...
                      IaIbdt2dW(1,1,end,i));
        if size(X0,4)>1
            switch compCase %spExp deviceExp
                case '00'
                    X3(:,:,1,i)=Exp(full(gather(Y)),X0(:,:,1,i));
                case '10'
                    X3(:,:,1,i)=Exp(gather(Y),X0(:,:,1,i));
                case '01'
                    X3(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0(:,:,1,i))));
                case '11'
                    X3(:,:,1,i)=Exp(Y,gpuArray(X0(:,:,1,i)));
            end
        else
            switch compCase %spExp deviceExp
                case '00'
                    X3(:,:,1,i)=Exp(full(gather(Y)),X0);
                case '10'
                    X3(:,:,1,i)=Exp(gather(Y),X0);
                case '01'
                    X3(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0)));
                case '11'
                    X3(:,:,1,i)=Exp(Y,gpuArray(X0));
            end
        end
    end
end
end