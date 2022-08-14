function X=magnusConst(A,B,X0,W,T,order)
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
N=size(W,3);
M=size(W,4);
dt=T/(N-1);
timegrid=reshape(linspace(0,T,N),1,1,[],1);

[spExp,deviceExp,deviceLog]=compMode(size(X0,1),size(X0,2),...
                                     issparse(A) && issparse(B));
compCase=sprintf('%d%d',spExp,deviceExp);

if size(X0,2)~=1
    X=zeros(size(A,1),size(A,1),1,M);
else
    X=zeros(size(A,1),1,1,M);
end
switch order
%     case 1
%         parfor i=1:1:M
%             
%         end
    case 2
        A2=A*A;
        BA=comm(B,A);
        IW=lebesgueInt(W,dt,T);
        if deviceLog
            A=gpuArray(A);
            B=gpuArray(B);
            A2=gpuArray(A2);
            BA=gpuArray(BA);
        end
        parfor i=1:M
            Y=secondorder(B,A,A2,BA,T,W(:,:,end,i),IW(:,:,end,i));
            if size(X0,4)>1
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0(:,:,1,i));
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0(:,:,1,i));
                    case '01'
                        X(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0(:,:,1,i))));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0(:,:,1,i)));
                end
            else
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0);
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0);
                    case '01'
                        X(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0)));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0));
                end
            end
        end
    case 3
        A2=A*A;
        BA=comm(B,A);
        BAA=comm(BA,A);
        BAB=comm(BA,B);
        IW=lebesgueInt(W,dt,T);
        IW2=lebesgueInt(W.^2,dt,T);
        IsW=lebesgueInt(timegrid.*W,dt,T);
        if deviceLog
            A=gpuArray(A);
            B=gpuArray(B);
            A2=gpuArray(A2);
            BA=gpuArray(BA);
            BAA=gpuArray(BAA);
            BAB=gpuArray(BAB);
        end
        parfor i=1:M
            Y=thirdorder(B,A,A2,BA,BAA,BAB,T,W(:,:,end,i),IW(:,:,end,i),IsW(:,:,end,i),IW2(:,:,end,i));
            if size(X0,4)>1
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0(:,:,1,i));
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0(:,:,1,i));
                    case '01'
                        X(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0(:,:,1,i))));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0(:,:,1,i)));
                end
            else
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0);
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0);
                    case '01'
                        X(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0)));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0));
                end
            end
        end
    otherwise
        error('Order %d not implemented',order)
end