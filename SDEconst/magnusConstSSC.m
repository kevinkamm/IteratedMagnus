function [X0,varargout]=magnusConstSSC(A,B,X0,W,T,dT,tol,region)
%%MAGNUS computes Ito-stochastic Magnus expansion for given matrix
% processes A_t=A.*a(t,Z_t), B.*b(t,Z_t), such that dX_t = B_t X_t dt + A_t X_t dW_t, X_0=X0.
% If X0 is a (d x 1) vector, expmv is used instead of expm for performance
% boost.
%
% Assumptions:
%   - Time grid is homogeneous.
%   - N is the number of time steps for Magnus logarithm and evaluation is
%     only at the terminal time
%   - Logarithm fully vectorized, small size of matrices
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
%   W (1 x 1 x N x M): 
%       Brownian motion
%   order (1 x 1 int): 
%       order of Magnus expansion
%   
d=size(A,1);
N=size(W,3);
M=size(W,4);
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
BAA=comm(BA,A);
BAB=comm(BA,B);

if deviceLog
    A=gpuArray(A);
    B=gpuArray(B);
    A2=gpuArray(A2);
    BA=gpuArray(BA);
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
            [X2,X3]=magnus23(X0(:,:,1,1:Mcut),W(1,1,ind,1:Mcut)-W(1,1,oldTi,1:Mcut),dt,t(currTi)-t(oldTi));
        else
            [X2,X3]=magnus23(X0,W(1,1,ind,1:Mcut)-W(1,1,oldTi,1:Mcut),dt,t(currTi)-t(oldTi));
        end
        currTol=isClose(X2,X3);
        if currTol>=1e9
            dTind=floor(dTind/2);
        end
%         step=floor(log10(currTol/tol));
        tempDt=floor(currDt/2);
%         tempDt2=floor(currDt/(2^(1+step)));
        if tempDt>=currDt || tempDt<=2
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
            X3=cat(4,X3,magnus3(X0(:,:,1,Mcut+1:end),W(1,1,ind,Mcut+1:end)-W(1,1,oldTi,Mcut+1:end),dt,t(currTi)-t(oldTi)));
        else
            X3=cat(4,X3,magnus3(X0,W(1,1,ind,Mcut+1:end)-W(1,1,oldTi,Mcut+1:end),dt,t(currTi)-t(oldTi)));
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
    err=mean(vecnorm(vecnorm(X3(region,region,1,:)-X2(region,region,1,:),2,1),2,2)./...
        vecnorm(vecnorm(X3(region,region,1,:),2,1),2,2),4);
%     err=mean(vecnorm(vecnorm(X3(region,region,1,:)-X2(region,region,1,:),2,1),2,2),4);
    if any(isnan(err),'all')
        err=1e9;
    end
end

function [X2,X3]=magnus23(X0,W,dt,T)
    currM=size(W,4);
    if size(X0,2)~=1
        X2=zeros(size(A,1),size(A,1),1,currM);
        X3=zeros(size(A,1),size(A,1),1,currM);
    else
        X2=zeros(size(A,1),1,1,currM);
        X3=zeros(size(A,1),1,1,currM);
    end
    timegrid=reshape(linspace(0,T,size(W,3)),1,1,[],1);
    IW=lebesgueInt(W,dt,T);
    IW2=lebesgueInt(W.^2,dt,T);
    IsW=lebesgueInt(timegrid.*W,dt,T);

    parfor i=1:currM
        Y2=secondorder(B,A,A2,BA,T,W(:,:,end,i),IW(:,:,end,i));
        Y3=thirdorder(B,A,A2,BA,BAA,BAB,T,W(:,:,end,i),IW(:,:,end,i),IsW(:,:,end,i),IW2(:,:,end,i));
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
function X3=magnus3(X0,W,dt,T)
    currM=size(W,4);
    if size(X0,2)~=1
        X3=zeros(size(A,1),size(A,1),1,currM);
    else
        X3=zeros(size(A,1),1,1,currM);
    end
    timegrid=reshape(linspace(0,T,size(W,3)),1,1,[],1);
    IW=lebesgueInt(W,dt,T);
    IW2=lebesgueInt(W.^2,dt,T);
    IsW=lebesgueInt(timegrid.*W,dt,T);

    parfor i=1:currM
        Y=thirdorder(B,A,A2,BA,BAA,BAB,T,W(:,:,end,i),IW(:,:,end,i),IsW(:,:,end,i),IW2(:,:,end,i));
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