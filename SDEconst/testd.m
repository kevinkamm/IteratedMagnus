clearvars;gpuDevice(1);
T=1;
dt=10^-3;
N=floor(T/dt)+1;
M=100;
d=300;
sigma=1/sqrt(10);
a=1.1;
xb=4;xa=-xb;
dx=(xb-xa)./(d+1);

% A=spdiags(randn(d,3),-1:1,d,d).*a./(2.*dx^2);
% B=spdiags(randn(d,3),-1:1,d,d).*sigma./(2.*dx);
A=randn(d,d);A=A./norm(A,'fro');
B=randn(d,d);B=B./norm(B,'fro');

t=reshape(linspace(0,T,N),1,1,[],1);
W=zeros(1,1,N,M);
W(1,1,2:end,:)=sqrt(dt).*randn(1,1,N-1,M);
W=cumsum(W,3);

X0=randn(d,d);

% loops=10;

% ctimeSPLog=0;
% for i=1:1:loops
%     ticSpLog=tic;
%     spLog(A,B,T,W,M,dt,t);
%     ctimeSPLog=ctimeSPLog+toc(ticSpLog);
% end
% ctimeSPLog=ctimeSPLog./loops

% ctimeGPUlog=0;
% for i=1:1:loops
%     ticGPUlog=tic;
%     gpuLog(A,B,T,W,M,dt,t);
%     ctimeGPUlog=ctimeGPUlog+toc(ticGPUlog);
% end
% ctimeGPUlog=ctimeGPUlog./loops

clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)

% timeit(@()spLog(A,B,T,W,M,dt,t))

gputimeit(@()gpuLog(A,B,T,W,M,dt,t,X0))
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)

timeit(@()cpuLog(A,B,T,W,M,dt,t,X0))



%%
% gputimeit(@()gpuVecLog(A,B,T,W,dt,t))
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)

% function gpuVecLog(A,B,T,W,dt,timegrid)
% doesnt work! 
%Error using gpuArray/bsxfun
%Maximum variable size allowed on the device is exceeded.
%
%     BA=comm(B,A);
%     BAA=comm(BA,A);
%     BAB=comm(BA,B);
%     BA=gpuArray(BA(:));
%     BAA=gpuArray(BAA(:));
%     BAB=gpuArray(BAB(:));
%     A2=A^2;A2=gpuArray(A2(:));
%     A=gpuArray(A(:));
%     B=gpuArray(B(:));
%     
%     W=gpuArray(reshape(W(1,1,end,:),1,[]));
%     IsW=gpuArray(reshape(lebesgueInt(W.*timegrid,dt,T),1,[]));
%     IW=gpuArray(reshape(lebesgueInt(W,dt,T),1,[]));
%     IW2=gpuArray(reshape(lebesgueInt(W.^2,dt,T),1,[]));
% %     parfor i=1:M
%     Otemp=B.*T+A.*W+...
%           -.5.*A2.*T+BA.*IW-.5.*BA.*T.*W+...
%           -(1/12).*BAB.*T.^2.*W+...
%           (1/12).*BAA.*T.*W.^2+...
%           BAB.*IsW-...
%           .5.*BAA.*W.*IW-...
%           .5.*BAB.*T.*IW+...
%           .5.*BAA.*IW2;
%     Otemp2=gather(Otemp);
% %     end
% end

function gpuLog(A,B,T,W,M,dt,timegrid,X0)
    BA=gpuArray(comm(B,A));
    BAA=gpuArray(comm(BA,A));
    BAB=gpuArray(comm(BA,B));
    A2=gpuArray(A^2);
    A=gpuArray(A);
    B=gpuArray(B);
    IsW=lebesgueInt(W.*timegrid,dt,T);
    IW=lebesgueInt(W,dt,T);
    IW2=lebesgueInt(W.^2,dt,T);
    parfor i=1:M
        Otemp=B.*T+A.*W(1,1,end,i)+...
              -.5.*A2.*T+BA.*IW(1,1,end,i)-.5.*BA.*T.*W(1,1,end,i)+...
              -(1/12).*BAB.*T.^2.*W(1,1,end,i)+...
              (1/12).*BAA.*T.*W(1,1,end,i).^2+...
              BAB.*IsW(1,1,end,i)-...
              .5.*BAA.*W(1,1,end,i).*IW(1,1,end,i)-...
              .5.*BAB.*T.*IW(1,1,end,i)+...
              .5.*BAA.*IW2(1,1,end,i);
        if size(X0,2)==1
            F=expmvtay2(Otemp,gpuArray(X0),30,'half');
        else
            F=expm(single(Otemp))*gpuArray(single(X0));
        end
        F=gather(F);
    end
end

function cpuLog(A,B,T,W,M,dt,timegrid,X0)
    BA=comm(B,A);
    BAA=comm(BA,A);
    BAB=comm(BA,B);
    A2=A^2;
    IsW=lebesgueInt(W.*timegrid,dt,T);
    IW=lebesgueInt(W,dt,T);
    IW2=lebesgueInt(W.^2,dt,T);
    parfor i=1:M
        Otemp=B.*T+A.*W(1,1,end,i)+...
              -.5.*A2.*T+BA.*IW(1,1,end,i)-.5.*BA.*T.*W(1,1,end,i)+...
              -(1/12).*BAB.*T.^2.*W(1,1,end,i)+...
              (1/12).*BAA.*T.*W(1,1,end,i).^2+...
              BAB.*IsW(1,1,end,i)-...
              .5.*BAA.*W(1,1,end,i).*IW(1,1,end,i)-...
              .5.*BAB.*T.*IW(1,1,end,i)+...
              .5.*BAA.*IW2(1,1,end,i);
        if size(X0,2)==1
            F=expmvtay2(Otemp,X0,30,'half');
        else
            F=expm(Otemp)*X0;
        end
        F=gather(F);
    end
end

% function spLog(A,B,T,W,M,dt,timegrid)
%     spMode=1;
%     O=firstorder(A,B,T,W,M,spMode)+...
%         secondorder(A,B,T,W,M,dt,spMode)+...
%         thirdorder(A,B,T,W,M,dt,timegrid,spMode);
% end

function O1=firstorder(A,B,T,W,M,spMode)
    if spMode
        O1=B(:).*T+...
            A(:).*reshape(W(1,1,end,:),1,M);
    else
        O1=B.*T+...
            A.*W(1,1,end,:);
    end
end
function O2=secondorder(A,B,T,W,M,dt,spMode)
    BA=comm(B,A);
    if spMode
        BA=BA(:);
        A2=A^2;
        A2=A2(:);
        IW=reshape(lebesgueInt(W,dt,T),1,M);
        O2=-A2./2.*T+...
            BA.*IW-BA.*T.*reshape(W(1,1,end,:),1,M)./2;
    else
        IW=lebesgueInt(W,dt,T);
        O2=-A^2./2.*T+...
            BA.*IW-BA.*T.*W(1,1,end,:)./2;
    end
end
function O3=thirdorder(A,B,T,W,M,dt,timegrid,spMode)
    BA=comm(B,A);
    BAA=comm(BA,A);
    BAB=comm(BA,B);
    if spMode
        BAA=BAA(:);
        BAB=BAB(:);
        IsW=reshape(lebesgueInt(W.*timegrid,dt,T),1,M);
        IW=reshape(lebesgueInt(W,dt,T),1,M);
        IW2=reshape(lebesgueInt(W.^2,dt,T),1,M);
        Wvec=reshape(W(1,1,end,:),1,M);
        O3=-BAB./12.*T.^2.*Wvec+...
          BAA./12.*T.*Wvec.^2+...
          BAB.*IsW-...
          BAA./2.*Wvec.*IW-...
          BAB./2.*T.*IW+...
          BAA./2.*IW2;
    else
        IsW=lebesgueInt(W.*timegrid,dt,T);
        IW=lebesgueInt(W,dt,T);
        IW2=lebesgueInt(W.^2,dt,T);
        O3=-BAB./12.*T.^2.*W(1,1,end,:)+...
          BAA./12.*T.*W(1,1,end,:).^2+...
          BAB.*IsW-...
          BAA./2.*W(1,1,end,:).*IW-...
          BAB./2.*T.*IW+...
          BAA./2.*IW2;
    end
end
function I=lebesgueInt(f,dt,T)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*T; 
    end
end

function C=comm(A,B)
    C=A*B-B*A;
end