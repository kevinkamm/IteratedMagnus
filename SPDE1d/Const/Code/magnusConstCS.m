function X=magnusConstCS(A,B,X0,W,T,dT,order)
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
NMagnusLog=size(W,3);
ind=unique([1:floor(NMagnusLog/round(T/dT)):NMagnusLog,NMagnusLog]);
for i=2:1:length(ind)
    WMagnusCurr=W(1,1,ind(i-1):ind(i),:)-W(1,1,ind(i-1),:);
    X=magnusConst(A,B,X0,WMagnusCurr,dT,order);
    X0=X;
end
end