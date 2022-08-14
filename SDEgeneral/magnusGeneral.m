function [X,Y]=magnusGeneral(A,B,X0,dW,T,order)
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
%   dW (1 x 1 x N x M): 
%       increments of the Brownian motion
%   order (1 x 1 int): 
%       order of Magnus expansion
%   
d=size(A,1);
N=size(dW,3)+1;
M=size(dW,4);
dt=T/(N-1);
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
% Y=zeros(d,d,1,M);
%% Compute Magnus Logarithm
%     ticLog=tic;
    switch order
        case 1
            Y=firstorder();
        case 2
            Y=firstorder()+secondorder();
%         case 3
%             Y=firstorder()+secondorder()+thirdorder();
        otherwise
            error('Order %d requested. Max order 2 allowed',order);
    end
%     ctimeLog=toc(ticLog);
%     fprintf('Elapsed time for logarithm: %g s\n',ctimeLog)
%% Compute Matrix Exponential
%     ticExp=tic;
    Y=gather(Y);
    X0=gather(X0);
    switch size(X0,2)
        case 0
            X=zeros(d,d,1,M);
            parfor i=1:M
                X(:,:,1,i)=expm(Y(:,:,1,i));
            end
        case 1
            X=zeros(d,1,1,M);
            if size(X0,4)>1
                parfor i=1:M
                    X(:,:,1,i)=expmvtay2(Y(:,:,1,i),X0(:,:,1,i),m,precision);
                end
            else
                parfor i=1:M
                    X(:,:,1,i)=expmvtay2(Y(:,:,1,i),X0,m,precision);
                end
            end
        case d
            X=zeros(d,d,1,M);
            if size(X0,4)>1
                parfor i=1:M
                    X(:,:,1,i)=expm(Y(:,:,1,i))*X0(:,:,1,i);
                end
            else
                parfor i=1:M
                    X(:,:,1,i)=expm(Y(:,:,1,i))*X0;
                end
            end
        otherwise
            error('Incompatible initial datum')
    end
%     ctimeExp=toc(ticExp);
%     fprintf('Elapsed time for exponential: %g s\n',ctimeExp)
%% expansion formulas
    function Y=firstorder()
        Y=lebesgueInt(B,dt,T)+stochInt(A,dW);
    end
    function Y=secondorder()
        IAdWs=cumstochInt(A,dW);
        IBds=cumlebesgueInt(B,dt,T);
        Y=(-lebesgueInt(pagemtimes(A,A),dt,T)+...
            stochInt(comm(A,IAdWs),dW)+...
            lebesgueInt(comm(B,IAdWs),dt,T)+...
            stochInt(comm(A,IBds),dW))./2+...
            lebesgueInt(comm(B,IBds),dt,T);
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