clearvars; close all; fclose('all'); rng(0);
availGPU=gpuDeviceCount("available");
if availGPU > 0
    device='gpu';
    gpu=gpuDevice(1);
else
    device='cpu';
end
% don't change, need multiprocessing for coefficients and multithreading
% for Magnus
delete(gcp('nocreate'));parpool('threads'); % multithreading
%% Parameters
% heat equation
% M=100;
% d=200;
% T=1;
% 
% dT=.005;
% dTSSC=2*dT;
% tol=1e-3;
% 
% dtMagnusLog=10^(-3);
% dtEuler=10^(-3);
% dtEulerRef=10^(-4);

% upper triangular
M=100;
d=100;
T=1;

dT=.05;

dtMagnusLog=10^(-4);
dtEuler=10^(-3);
dtEulerRef=10^(-4);

dWEulerTotal={};
aTotal={};
bTotal={};

NMagnusLog= floor(T/dtMagnusLog)+1;
NEulerRef= floor(T/dtEulerRef)+1;
NEuler= floor(T/dtEuler)+1;

t=linspace(0,T,NEulerRef);
[dWvec,tIndvec]=brownianIncrement(T,[NEulerRef,NMagnusLog,NEuler],M);
dWEulerRef=dWvec{1};
dWMagnus=dWvec{2};
dWEuler=dWvec{3};
tInd=tIndvec{1};
tInd2=tIndvec{2};

method='Langevin1';

ticCoeff=tic;
[A,a,B1,B2,b,X0,XExact,region1,region2,EulerLangevin]=coefficients(t,M,d,method);
ctimeCoeff=toc(ticCoeff);

if size(a,3)>1
    amagnus=a(:,:,tInd,:);
    bmagnus=b(:,:,tInd,:);
    aEuler=a(:,:,tInd2,:);
    bEuler=b(:,:,tInd2,:);
else
    amagnus=a;
    bmagnus=b;
    aEuler=a;
    bEuler=b;
end

%% Magnus
disp('Compute Magnus with constant step size')
% Constant step size
ticMagnus2CS=tic;
X2CS=magnusSeparableCS(A,amagnus,B1,B2,bmagnus,X0,dWMagnus,T,dT,2);
ctimeMagnus2CS=toc(ticMagnus2CS);
X2CS=reshape(X2CS,d,d,1,M);
fprintf('Elapsed time iterated Magnus 2: %g s\n',ctimeMagnus2CS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)
    
% ticMagnus3CS=tic;
% X3CS=magnusSeparableCS(A,amagnus,B1,B2,bmagnus,X0,dWMagnus,T,dT,3);
% ctimeMagnus3CS=toc(ticMagnus3CS);
% X3CS=reshape(X3CS,d,d,1,M);
% fprintf('Elapsed time iterated Magnus 3: %g s\n',ctimeMagnus3CS);
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)

%% Euler
disp('Compute Euler reference')

ticEulerRef=tic;
XEulerRef=eulerSeparableLangevin(T,dWEulerRef,a,b,EulerLangevin);
ctimeEulerRef=toc(ticEulerRef);
fprintf('Elapsed time EulerRef: %g s\n',ctimeEulerRef);
%%
disp('Compute Euler')
ticEuler=tic;
XEuler=eulerSeparableLangevin(T,dWEuler,aEuler,bEuler,EulerLangevin);
ctimeEuler=toc(ticEuler);
fprintf('Elapsed time Euler: %g s\n',ctimeEuler)

%% Errors

relErrEulerRefEuler=meanRelError(XEulerRef(region1,region2,1,:),XEuler(region1,region2,1,:));
% relErrEulerRefMCS3=meanRelError(XEulerRef(region1,region2,1,:),X3CS(region1,region2,1,:));
relErrEulerRefMCS2=meanRelError(XEulerRef(region1,region2,1,:),X2CS(region1,region2,1,:));

fprintf('EulerRef vs Euler : %3.3e\n',relErrEulerRefEuler)
% fprintf('EulerRef vs iterated Magnus 3: %3.3e\n',relErrEulerRefMCS3)
fprintf('EulerRef vs iterated Magnus 2: %3.3e\n',relErrEulerRefMCS2)


%% Save results
if strcmp(method,'Tridiag2x2exm1')
    fileName=[method,sprintf('_T%1.2f_d2_dtEulerRef%1.3e_dtEuler%1.3e_dtLog%1.3e_dTSSC%1.3e_dTiter%1.3e_M%d',T,dtEulerRef,dtEuler,dtMagnusLog,dTSSC,dT,M)];
else
    fileName=[method,sprintf('_T%1.2f_d%d_dtEulerRef%1.3e_dtEuler%1.3e_dtLog%1.3e_dTSSC%1.3e_dTiter%1.3e_M%d',T,d,dtEulerRef,dtEuler,dtMagnusLog,dTSSC,dT,M)];
end
matDir=['Results/Mat','/',method];
outputMatlab(matDir,fileName,...
              T,dtEulerRef,dtEuler,dtMagnusLog,dT,M,region1,d,...
              ctimeEulerRef,ctimeEuler,ctimeMagnus2CS,[],...
              relErrEulerRefMCS2,[],...
              relErrEulerRefEuler)


