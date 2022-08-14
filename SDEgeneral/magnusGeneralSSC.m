function [X0,varargout]=magnusGeneralSSC(A,B,X0,dW,T,dT,tol)
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
N=size(dW,3)+1;
M=size(dW,4);
Mcut=ceil(M/10);
% dt=T/(N-1);
m=30;
precision = 'half';

if ~isgpuarray(A)
    A=gpuArray(single(A));
end
if ~isgpuarray(B)
    B=gpuArray(single(B));
end
if ~isgpuarray(dW)
    dW=gpuArray(single(dW));
end

stepSizes=[];
t=linspace(0,T,N);
oldT=0;
currT=0;
oldTi=1;
while(currT<T)
    currDt=min(dT,T-currT);
    X2=1;X3=1e9;
    n=0;
    while(isClose(X2,X3)>tol && n<10)
        currT=oldT+currDt;
%         fprintf('Old T %g + Curr Step %g = Curr T %g (n=%d)\n',oldT,currDt,currT,n);
        currTi=find(t<=currT,1,'last');
        ind=oldTi:currTi;
        dt=(currT-oldT)/(length(ind)-1);
%         tic;
        if size(A,3)>1
            if size(X0,4)>1
                [X2,X3]=magnusLogStep(X0(:,:,1,1:Mcut),A(:,:,ind,1:Mcut),B(:,:,ind,1:Mcut),dW(1,1,ind(1:end-1),1:Mcut),dt,currT-oldT);
            else
                [X2,X3]=magnusLogStep(X0,A(:,:,ind,1:Mcut),B(:,:,ind,1:Mcut),dW(1,1,ind(1:end-1),1:Mcut),dt,currT-oldT);
            end
        else
            if size(X0,4)>1
                [X2,X3]=magnusLogStep(X0(:,:,1,1:Mcut),A,B,dW(1,1,ind(1:end-1),1:Mcut),dt,currT-oldT);
            else
                [X2,X3]=magnusLogStep(X0,A,B,dW(1,1,ind(1:end-1),1:Mcut),dt,currT-oldT);
            end
        end
%         toc;
        currDt=currDt/2;
        n=n+1;
    end
    if Mcut<M
        if size(A,3)>1
            if size(X0,4)>1
                X3=cat(4,X3,magnusLog3(X0(:,:,1,Mcut+1:end),A(:,:,ind,Mcut+1:end),B(:,:,ind,Mcut+1:end),dW(1,1,ind(1:end-1),Mcut+1:end),dt,currT-oldT));
            else
                X3=cat(4,X3,magnusLog3(X0,A(:,:,ind,Mcut+1:end),B(:,:,ind,Mcut+1:end),dW(1,1,ind(1:end-1),Mcut+1:end),dt,currT-oldT));
            end
        else
            if size(X0,4)>1
                X3=cat(4,X3,magnusLog3(X0(:,:,1,Mcut+1:end),A(:,:,1,Mcut+1:end),B(:,:,1,Mcut+1:end),dW(1,1,ind(1:end-1),Mcut+1:end),dt,currT-oldT));
            else
                X3=cat(4,X3,magnusLog3(X0,A,B,dW(1,1,ind(1:end-1),Mcut+1:end),dt,currT-oldT));
            end
        end
    end
    oldT=currT;
    oldTi=currTi;
    if n>10
        warning('Number of step size reduction exceeded. Bad approximation possible.')
    end
    stepSizes(end+1)=currDt*2;
    X0=X3;
end
if nargout>1
    varargout{1}=stepSizes;
end

function err=isClose(Y2,Y3)
    err=mean(vecnorm(vecnorm(Y3-Y2,2,1),2,2)./...
        vecnorm(vecnorm(Y3,2,1),2,2),4);
%     err=mean(sum(abs(Y3-Y2),[1,2]),4);
end
function [X2,X3]=magnusLogStep(X0,a,b,dW,dt,T)
    Y2=firstorder(a,b,dW,dt,T)+secondorder(a,b,dW,dt,T);
    Y3=Y2+thirdorder(a,b,dW,dt,T);
    currM=size(dW,4);
    if isempty(X0)
        X2=zeros(d,d,1,currM);
        X3=zeros(d,d,1,currM);
    else
        X2=zeros(size(X0));
        X3=zeros(size(X0));
    end
    switch size(X0,2)
        case 0
            parfor i=1:currM
                X2(:,:,1,i)=expm(Y2(:,:,1,i));
                X3(:,:,1,i)=expm(Y3(:,:,1,i));
            end
        case 1
            if size(X0,4)>1
                parfor i=1:currM
                    X2(:,:,1,i)=expmvtay2(Y2(:,:,1,i),X0(:,:,1,i),m,precision);
                    X3(:,:,1,i)=expmvtay2(Y3(:,:,1,i),X0(:,:,1,i),m,precision);
                end
            else
                parfor i=1:currM
                    X2(:,:,1,i)=expmvtay2(Y2(:,:,1,i),X0,m,precision);
                    X3(:,:,1,i)=expmvtay2(Y3(:,:,1,i),X0,m,precision);
                end
            end
        case d
            if size(X0,4)>1
                parfor i=1:currM
                    X2(:,:,1,i)=expm(Y2(:,:,1,i))*X0(:,:,1,i);
                    X3(:,:,1,i)=expm(Y3(:,:,1,i))*X0(:,:,1,i);
                end
            else
                parfor i=1:currM
                    X2(:,:,1,i)=expm(Y2(:,:,1,i))*X0;
                    X3(:,:,1,i)=expm(Y3(:,:,1,i))*X0;
                end
            end
        otherwise
            error('Incompatible initial datum')
    end
end
function X3=magnusLog3(X0,a,b,dW,dt,T)
    Y3=firstorder(a,b,dW,dt,T)+secondorder(a,b,dW,dt,T)+thirdorder(a,b,dW,dt,T);
    currM=size(dW,4);
    if isempty(X0)
        X3=zeros(d,d,1,currM);
    else
        X3=zeros(size(X0));
    end
    switch size(X0,2)
        case 0
            parfor i=1:currM
                X3(:,:,1,i)=expm(Y3(:,:,1,i));
            end
        case 1
            if size(X0,4)>1
                parfor i=1:currM
                    X3(:,:,1,i)=expmvtay2(Y3(:,:,1,i),X0(:,:,1,i),m,precision);
                end
            else
                parfor i=1:currM
                    X3(:,:,1,i)=expmvtay2(Y3(:,:,1,i),X0,m,precision);
                end
            end
        case d
            if size(X0,4)>1
                parfor i=1:currM
                    X3(:,:,1,i)=expm(Y3(:,:,1,i))*X0(:,:,1,i);
                end
            else
                parfor i=1:currM
                    X3(:,:,1,i)=expm(Y3(:,:,1,i))*X0;
                end
            end
        otherwise
            error('Incompatible initial datum')
    end
end

%% expansion formulas
    function Y=firstorder(A,B,dW,dt,T)
        Y=lebesgueInt(B,dt,T)+stochInt(A,dW);
    end
    function Y=secondorder(A,B,dW,dt,T)
        IAdWs=cumstochInt(A,dW);
        IBds=cumlebesgueInt(B,dt,T);
        Y=(-lebesgueInt(pagemtimes(A,A),dt,T)+...
            stochInt(comm(A,IAdWs),dW)+...
            lebesgueInt(comm(B,IAdWs),dt,T)+...
            stochInt(comm(A,IBds),dW))./2+...
            lebesgueInt(comm(B,IBds),dt,T);
    end
    function Y=thirdorder(A,B,dW,dt,T)
        IAdWs=cumstochInt(A,dW);
        IBds=cumlebesgueInt(B,dt,T);
        AIAdWs=comm(A,IAdWs);
        BIAdWs=comm(B,IAdWs);
        AIBds=comm(A,IBds);
        BIBds=comm(B,IBds);

        Y=-lebesgueInt(comm(A,AIAdWs),dt,T)./6-...
           stochInt(comm(IAdWs,AIAdWs),dW)./12-...
           lebesgueInt(comm(A,AIBds),dt,T)./6-...
           lebesgueInt(comm(IAdWs,BIAdWs),dt,T)./12-...
           stochInt(comm(IBds,AIAdWs)+comm(IAdWs,AIBds),dW)./6-...
           stochInt(comm(IBds,AIBds),dW)./12-...
           lebesgueInt(comm(IBds,BIAdWs)+comm(IAdWs,BIBds),dt,T)./12-...
           lebesgueInt(comm(IBds,AIBds),dt,T)./6;
    end
end
function I=lebesgueInt(f,dt,T)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*T; 
    end
    I=gather(I);
end
function I=stochInt(f,dW)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:).*dW,3); 
    else
        I=f.*sum(dW,3); 
    end
    I=gather(I);
end
function I=cumlebesgueInt(f,dt,T)
    I=zeros(size(f));
    if size(f,3)>1
        I(:,:,2:end,:)=cumsum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*reshape(linspace(0,T,round(T/dt)+1),1,1,[],1); 
    end
end
function I=cumstochInt(f,dW)
    I=zeros(size(f,1),size(f,2),size(dW,3)+1,size(dW,4));
    if size(f,3)>1
        I(:,:,2:end,:)=cumsum(f(:,:,1:1:end-1,:).*dW,3); 
    else
        I(:,:,2:end,:)=f.*cumsum(dW,3); 
    end
end
function C=comm(A,B)
    C=pagemtimes(A,B)-pagemtimes(B,A);
end