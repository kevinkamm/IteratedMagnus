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
M=1000;
d=2;
T=1;

dT=.1;
dTSSC=2*dT;
tol=1e-3;

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

method='Tridiag2x2exm1';
% method='HeatEquation1';
% method='Langevin1';

ticCoeff=tic;
[A,a,B,b,X0,XExact,region1,region2,EulerLangevin]=coefficients(t,M,d,dWEulerRef,method);
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
X2CS=magnusSeparableCS(A,amagnus,B,bmagnus,X0,dWMagnus,T,dT,2);
ctimeMagnus2CS=toc(ticMagnus2CS);
fprintf('Elapsed time iterated Magnus 2: %g s\n',ctimeMagnus2CS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)
    
ticMagnus3CS=tic;
X3CS=magnusSeparableCS(A,amagnus,B,bmagnus,X0,dWMagnus,T,dT,3);
ctimeMagnus3CS=toc(ticMagnus3CS);
fprintf('Elapsed time iterated Magnus 3: %g s\n',ctimeMagnus3CS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)
%% Single step
disp('Compute Magnus with one step')
ticMagnus2SS=tic;
X2SS=magnusSeparable(A,amagnus,B,bmagnus,X0,dWMagnus,T,2);
ctimeMagnus2SS=toc(ticMagnus2SS);
fprintf('Elapsed time single step Magnus 2: %g s\n',ctimeMagnus2SS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)

ticMagnus3SS=tic;
X3SS=magnusSeparable(A,amagnus,B,bmagnus,X0,dWMagnus,T,3);
ctimeMagnus3SS=toc(ticMagnus3SS);
fprintf('Elapsed time single step Magnus 3: %g s\n',ctimeMagnus3SS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)
%% Step size control
% disp('Compute Magnus with step size control')
% ticMagnus3SSC=tic;
% [X3SSC,stepsizes]=magnusSeparableSSC(A,amagnus,B,bmagnus,X0,dWMagnus,T,dTSSC,tol);
% ctimeMagnus3SSC=toc(ticMagnus3SSC);
% fprintf('Elapsed time step size control Magnus 3: %g s\n',ctimeMagnus3SSC);
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)
% if ~isempty(EulerLangevin)
%     X3SSC=reshape(X3SSC,d,d,1,M);
%     X3CS=reshape(X3CS,d,d,1,M);
%     X2CS=reshape(X2CS,d,d,1,M);
%     X3SS=reshape(X3SS,d,d,1,M);
%     X2SS=reshape(X2SS,d,d,1,M);
% end
%% Euler
disp('Compute Euler reference')
if isempty(EulerLangevin)
    A=full(A);
    B=full(B);
end
ticEulerRef=tic;
if isempty(EulerLangevin)
    XEulerRef=eulerSeparable(A,a,B,b,X0,dWEulerRef,T);
else
    XEulerRef=eulerSeparableLangevin(T,dWeulerRef,a,b,EulerLangevin);
end
ctimeEulerRef=toc(ticEulerRef);
fprintf('Elapsed time EulerRef: %g s\n',ctimeEulerRef);
%%
disp('Compute Euler')
ticEuler=tic;
if isempty(EulerLangevin)
    XEuler=eulerSeparable(A,aEuler,B,bEuler,X0,dWEuler,T);
else
    XEuler=eulerSeparableLangevin(T,dWeuler,aEuler,bEuler,EulerLangevin);
end
ctimeEuler=toc(ticEuler);
fprintf('Elapsed time Euler: %g s\n',ctimeEuler)

%% Errors
if ~isempty(XExact)
    relErrExactEulerRef=meanRelError(XExact(region1,region2,1,:),XEulerRef(region1,region2,1,:));
    relErrExactEuler=meanRelError(XExact(region1,region2,1,:),XEuler(region1,region2,1,:));
%     relErrExactMSSC=meanRelError(XExact(region1,region2,1,:),X3SSC(region1,region2,1,:));
    relErrExactMCS3=meanRelError(XExact(region1,region2,1,:),X3CS(region1,region2,1,:));
    relErrExactMCS2=meanRelError(XExact(region1,region2,1,:),X2CS(region1,region2,1,:));
    relErrExactMSS3=meanRelError(XExact(region1,region2,1,:),X3SS(region1,region2,1,:));
    relErrExactMSS2=meanRelError(XExact(region1,region2,1,:),X2SS(region1,region2,1,:));
    fprintf('Exact vs EulerRef : %3.3e\n', relErrExactEulerRef)
    fprintf('Exact vs Euler : %3.3e\n',relErrExactEuler)
%     fprintf('Exact vs SSC Magnus : %3.3e\n',relErrExactMSSC)
    fprintf('Exact vs iterated Magnus 3: %3.3e\n',relErrExactMCS3)
    fprintf('Exact vs iterated Magnus 2: %3.3e\n',relErrExactMCS2)
    fprintf('Exact vs single Magnus 3: %3.3e\n',relErrExactMSS3)
    fprintf('Exact vs single Magnus 2: %3.3e\n',relErrExactMSS2)
else
    relErrEulerRefEuler=meanRelError(XEulerRef(region1,region2,1,:),XEuler(region1,region2,1,:));
%     relErrEulerRefMSSC=meanRelError(XEulerRef(region1,region2,1,:),X3SSC(region1,region2,1,:));
    relErrEulerRefMCS3=meanRelError(XEulerRef(region1,region2,1,:),X3CS(region1,region2,1,:));
    relErrEulerRefMCS2=meanRelError(XEulerRef(region1,region2,1,:),X2CS(region1,region2,1,:));
    relErrEulerRefMSS3=meanRelError(XEulerRef(region1,region2,1,:),X3SS(region1,region2,1,:));
    relErrEulerRefMSS2=meanRelError(XEulerRef(region1,region2,1,:),X2SS(region1,region2,1,:));
    fprintf('EulerRef vs Euler : %3.3e\n',relErrEulerRefEuler)
%     fprintf('EulerRef vs SSC Magnus : %3.3e\n',relErrEulerRefMSSC)
    fprintf('EulerRef vs iterated Magnus 3: %3.3e\n',relErrEulerRefMCS3)
    fprintf('EulerRef vs iterated Magnus 2: %3.3e\n',relErrEulerRefMCS2)
    fprintf('EulerRef vs single Magnus 3: %3.3e\n',relErrEulerRefMSS3)
    fprintf('EulerRef vs single Magnus 2: %3.3e\n',relErrEulerRefMSS2)
end
%% Save results
if strcmp(method,'Tridiag2x2exm1')
    fileName=[method,sprintf('_T%1.2f_d2_dtEulerRef%1.3e_dtLog%1.3e_dTSSC%1.3e_dTiter%1.3e_M%d',T,dtEulerRef,dtMagnusLog,dTSSC,dT,M)];
else
    fileName=[method,sprintf('_T%1.2f_d%d_dtEulerRef%1.3e_dtLog%1.3e_dTSSC%1.3e_dTiter%1.3e_M%d',T,d,dtEulerRef,dtMagnusLog,dTSSC,dT,M)];
end
matDir=['Results/Mat','/',method];
if ~exist(matDir,'dir')
    mkdir(matDir);
end
save([matDir,'/',fileName,'.mat'],'relErr*')
save([matDir,'/',fileName,'.mat'],'ctime*','-append')


