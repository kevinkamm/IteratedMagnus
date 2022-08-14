function X=eulerGeneral(A,B,X0,dW,T)
%%MAGNUS computes Euler Maruyama scheme for given matrix
% processes A_t, B_t, such that dX_t = B_t X_t dt + A_t X_t dW_t, X_0=X0.
% If X0 is a (d x 1) vector, matrix vector multiplication is used for 
% performance boost.
%
% Assumptions:
%   - Time grid is homogeneous.
%   - Value only at terminal time
%
% Input:
%   A (d x d x N x M array): 
%       matrix process at time t and trajectory w
%   B (d x d x N x M array): 
%       matrix process at time t and trajectory w
%   X0 (empty, d x 1 x 1 x (M), d x d x 1 x (M) array): 
%       initial value, if X0=[] it is assumed to be the identity matrix and
%       last axis is optional for random initial datum
%   dW (1 x 1 x N x M): 
%       increments of the Brownian motion
% 
d=size(A,1);
N=size(dW,3)+1;
dt=T/(N-1);

if isempty(X0)
    X0=eye(d);
end

X=X0;
if size(B,3)>1
    for i=1:1:N-1
        X=X+pagemtimes(B(:,:,i,:),X).*dt+pagemtimes(A(:,:,i,:),X).*dW(1,1,i,:);
    end
else
    for i=1:1:N-1
        X=X+pagemtimes(B,X).*dt+pagemtimes(A.*a,X).*dW(1,1,i,:);
    end
end

end