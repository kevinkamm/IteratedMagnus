function X=magnusSeparable(A,a,B,b,X0,dW,T,order)
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
        IadW=cumstochInt(a,dW);
        Ibdt=cumlebesgueInt(b,dt,T);
        Ia2dt=lebesgueInt(a.^2,dt,T);
        IbIadWdt=lebesgueInt(b.*IadW,dt,T);
        IaIbdtdW=stochInt(a.*Ibdt,dW);
        if deviceLog
            A=gpuArray(A);
            B=gpuArray(B);
            A2=gpuArray(A2);
            BA=gpuArray(BA);
        end
        parfor i=1:M
            Y=secondorder(B,A,A2,BA,Ibdt(1,1,end,i),IadW(1,1,end,i),Ia2dt(1,1,end,i),IbIadWdt(1,1,end,i),IaIbdtdW(1,1,end,i));
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
        BA2=comm(B,A2);
        BAA=comm(BA,A);
        BAB=comm(BA,B);


        Ibdt=cumlebesgueInt(b,dt,T);
        IadW=cumstochInt(a,dW);
        Ia2dt=cumlebesgueInt(a.^2,dt,T);
        IbIadWdt=cumlebesgueInt(b.*IadW,dt,T);
        IaIbdtdW=cumstochInt(a.*Ibdt,dW);
        Ia2Ibdtdt=lebesgueInt(a.^2.*Ibdt,dt,T);
        IbIa2dtdt=lebesgueInt(b.*Ia2dt,dt,T);
        IIadWIbdtadW=stochInt(IadW.*Ibdt.*a,dW);
        IbIadW2dt=lebesgueInt(b.*(IadW).^2,dt,T);
        IaIaIbdtdWdW=stochInt(a.*IaIbdtdW,dW);
        IaIbIadWdtdt=lebesgueInt(a.*IbIadWdt,dt,T);
        IbIbdtIadWdt=lebesgueInt(b.*Ibdt.*IadW,dt,T);
        IbIbIadWdtdt=lebesgueInt(b.*IbIadWdt,dt,T);
        IaIbdtdt=cumlebesgueInt(a.*Ibdt,dt,T);
        IbIaIbdtdtdt=lebesgueInt(b.*IaIbdtdt,dt,T);
        IaIbdt2dW=stochInt(a.*(Ibdt).^2,dW);
        if deviceLog
            A=gpuArray(A);
            B=gpuArray(B);
            A2=gpuArray(A2);
            BA=gpuArray(BA);
            BA2=gpuArray(BA2);
            BAA=gpuArray(BAA);
            BAB=gpuArray(BAB);
        end
        parfor i=1:M
            Y=thirdorder(B,A,A2,BA,BA2,BAB,BAA,...
                      Ibdt(1,1,end,i),IadW(1,1,end,i),Ia2dt(1,1,end,i),IbIadWdt(1,1,end,i),IaIbdtdW(1,1,end,i),...
                      Ia2Ibdtdt(1,1,end,i),IbIa2dtdt(1,1,end,i),IIadWIbdtadW(1,1,end,i),IbIadW2dt(1,1,end,i),IaIaIbdtdWdW(1,1,end,i),...
                      IaIbIadWdtdt(1,1,end,i),IbIbdtIadWdt(1,1,end,i),IbIbIadWdtdt(1,1,end,i),IbIaIbdtdtdt(1,1,end,i),...
                      IaIbdt2dW(1,1,end,i));
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