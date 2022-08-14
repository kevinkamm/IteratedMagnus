function [dWvec,tIndvec]=brownianIncrement(T,Nvec,M)
%%BROWNIANMOTION computes the Brownian motion with different time steps for
% Euler and Magnus
%   Input:
%       T (1 x 1 double): 
%       	the finite time horizon
%       Nvec (n x 1 array): 
%           the number of time-steps 
%       M (1 x 1 int): 
%           the number of simulations
%   Output:
%       dWvec (n x 1 cell with 1 x 1 x Nvec(i) x M entries): 
%           increment of BM for Euler
%       tIndvec (n x 1 cell): 
%           indices to match Euler and Magnus time grids

    [Nvec,sortInd] = sort(Nvec,2,"descend");
    Nmax = Nvec(1);
    tIndvec=cell(length(Nvec)-1,1);
    dWvec=cell(length(Nvec),1);

    dt = T/(Nmax(1)-1);

    dW =sqrt(dt).*randn(1,1,Nmax-1,M);
    dWvec{1}=dW;

    W=zeros(1,1,Nmax,M);
    W(1,1,2:end,:)=dW;
    W=cumsum(W,3);
    
    for i=2:length(Nvec)
        N=Nvec(i);
        [dWcoarse,tInd]=coarseBM(N);
        dWvec{i}=dWcoarse;
        tIndvec{i-1}=tInd;
    end
    dWvec=dWvec(sortInd);
    iMax=find(Nmax==Nvec,1,"first");
    tIndvec=tIndvec(sortInd([1:iMax-1,iMax+1:length(sortInd)])-1);
    function [dWcoarse,tInd]=coarseBM(N)
        tInd=1:1:N;
        tInd(2:1:end)=tInd(1:1:end-1).*floor((Nmax-1)/(N-1))+1;
        if mod((Nmax-1),(N-1))~=0
            error('Sub time grids incompatible')
        end
        Wcoarse=W(1,1,tInd,:);
        dWcoarse=diff(Wcoarse,1,3);
    end
end
