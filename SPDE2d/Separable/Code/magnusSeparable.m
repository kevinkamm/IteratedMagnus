function X=magnusSeparable(A,a,B1,B2,b,X0,dW,T,order)
%%MAGNUSSEPARABLE computes Ito-stochastic Magnus expansion for given matrix
% processes A_t=A.*a(t,Z_t), B.*b(t,Z_t), such that dX_t = B_t X_t dt + A_t X_t dW_t, X_0=X0.
% If X0 is a (d x 1) vector, expmv is used instead of expm for performance
% boost. This is a single step method.
%
% Assumptions:
%   - Time grid is homogeneous.
%   - N is the number of time steps for Magnus logarithm and evaluation is
%     only at the terminal time
%
% Input:
%   A (d x d x 1 x 1 array): 
%       constant matrix 
%   a (1 x 1 x N x M array): 
%       1 dimensional stochastic process
%   B (d x d x 1 x 1 array): 
%       constant matrix
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
%       constant step size
%   order (1 x 1 int): 
%       order of Magnus expansion
%
% Output:
%   X (d x d x 1 x M array):
%       contains the solution at t=T
%
% See also: magnusSeparableCS, magnusSeparableSSC
N=size(dW,3)+1;
M=size(dW,4);
dt=T/(N-1);
t=linspace(0,T,N);
[spExp,deviceExp,deviceLog]=compMode(size(X0,1),size(X0,2),...
                                     issparse(A) && issparse(B1));
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
        B1A=comm(B1,A);
        B2A=comm(B2,A);
        B1B2=comm(B1,B2);

        Ibdt=cumlebesgueInt(b,dt,T);
        IadW=cumstochInt(a,dW);

        Ia2dt=lebesgueInt(a.^2,dt,T);
        IbIadWdt=lebesgueInt(b.*IadW,dt,T);
        IIadWdt=lebesgueInt(IadW,dt,T);
        IIbdtdt=lebesgueInt(Ibdt,dt,T);
        Isbdt=lebesgueInt(t.*b,dt,T);
        IaIbdtdW=stochInt(a.*Ibdt,dW);
        IsadW=stochInt(t.*a,dW);

        if deviceLog
            A=gpuArray(A);
            B1=gpuArray(B1);
            B2=gpuArray(B2);
            A2=gpuArray(A2);
            B1A=gpuArray(B1A);
            B2A=gpuArray(B2A);
            B1B2=gpuArray(B1B2);
        end
        parfor i=1:M
%             Y=secondorder(B,A,A2,BA,Ibdt(1,1,end,i),IadW(1,1,end,i),Ia2dt(1,1,end,i),IbIadWdt(1,1,end,i),IaIbdtdW(1,1,end,i));
            Y=secondorder(B1,B2,A,A2,B1A,B2A,B1B2,T,Ibdt(1,1,end,i),IadW(1,1,end,i),Ia2dt(1,1,end,i),IbIadWdt(1,1,end,i),IIadWdt(1,1,end,i),IaIbdtdW(1,1,end,i),IsadW(1,1,end,i),IIbdtdt(1,1,end,i),Isbdt(1,1,end,i));
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
end