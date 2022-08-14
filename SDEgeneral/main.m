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
M=1000;
d=3;
T=1;

dT=.1;
dTSSC=.05;
tol=1e-5;

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

% method='BMsig';
% method='Const';
method='Tridiag2x2exm1';

ticCoeff=tic;
[A,B,X0,XExact,region1,region2]=coefficients(t,M,d,dWEulerRef,method);
ctimeCoeff=toc(ticCoeff);

if size(A,3)>1
    Amagnus=A(:,:,tInd,:);
    Bmagnus=B(:,:,tInd,:);
    AEuler=A(:,:,tInd2,:);
    BEuler=B(:,:,tInd2,:);
else
    Amagnus=A;
    Bmagnus=B;
    AEuler=A;
    BEuler=B;
end

%% Magnus
% Constant step size
ticMagnus2CS=tic;
X2CS=magnusGeneralCS(Amagnus,Bmagnus,X0,dWMagnus,T,dT,2);
ctimeMagnus2CS=toc(ticMagnus2CS);
fprintf('Elapsed time iterated Magnus 2: %g s\n',ctimeMagnus2CS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)

% ticMagnus3CS=tic;
% X3CS=magnusGeneralCS(Amagnus,Bmagnus,X0,dWMagnus,T,dT,3);
% ctimeMagnus3CS=toc(ticMagnus3CS);
% fprintf('Elapsed time iterated Magnus 3: %g s\n',ctimeMagnus3CS);
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)
%% Single step
ticMagnus2SS=tic;
X2SS=magnusGeneral(Amagnus,Bmagnus,X0,dWMagnus,T,2);
ctimeMagnus2SS=toc(ticMagnus2SS);
fprintf('Elapsed time single step Magnus 2: %g s\n',ctimeMagnus2SS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)

% ticMagnus3SS=tic;
% X3SS=magnusGeneral(Amagnus,Bmagnus,X0,dWMagnus,T,3);
% ctimeMagnus3SS=toc(ticMagnus3SS);
% fprintf('Elapsed time single step Magnus 3: %g s\n',ctimeMagnus3SS);
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)

%% Step size control
% ticMagnus3SSC=tic;
% [X3SSC,stepsizes]=magnusGeneralSSC(Amagnus,Bmagnus,X0,dWMagnus,T,dTSSC,tol);
% ctimeMagnus3SSC=toc(ticMagnus3SSC);
% fprintf('Elapsed time step size control Magnus 3: %g s\n',ctimeMagnus3SSC);
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)
%% Euler
ticEulerRef=tic;
XEulerRef=eulerGeneral(A,B,X0,dWEulerRef,T);
ctimeEulerRef=toc(ticEulerRef);
fprintf('Elapsed time EulerRef: %g s\n',ctimeEulerRef);

ticEuler=tic;
XEuler=eulerGeneral(AEuler,BEuler,X0,dWEuler,T);
ctimeEuler=toc(ticEuler);
fprintf('Elapsed time Euler: %g s\n',ctimeEuler)

%% Errors
if ~isempty(XExact)
    relErrExactEulerRef=meanRelError(XExact(region1,region2,end,:),XEulerRef(region1,region2,end,:));
    relErrExactEuler=meanRelError(XExact(region1,region2,end,:),XEuler(region1,region2,end,:));
%     relErrExactMSSC=meanRelError(XExact(region1,region2,end,:),X3SSC(region1,region2,end,:));
%     relErrExactMCS3=meanRelError(XExact(region1,region2,end,:),X3CS(region1,region2,end,:));
    relErrExactMCS2=meanRelError(XExact(region1,region2,end,:),X2CS(region1,region2,end,:));
%     relErrExactMSS3=meanRelError(XExact(region1,region2,end,:),X3SS(region1,region2,end,:));
    relErrExactMSS2=meanRelError(XExact(region1,region2,end,:),X2SS(region1,region2,end,:));
    fprintf('Exact vs EulerRef : %3.3e\n', relErrExactEulerRef)
    fprintf('Exact vs Euler : %3.3e\n',relErrExactEuler)
%     fprintf('Exact vs SSC Magnus : %3.3e\n',relErrExactMSSC)
%     fprintf('Exact vs iterated Magnus 3: %3.3e\n',relErrExactMCS3)
    fprintf('Exact vs iterated Magnus 2: %3.3e\n',relErrExactMCS2)
%     fprintf('Exact vs single Magnus 3: %3.3e\n',relErrExactMSS3)
    fprintf('Exact vs single Magnus 2: %3.3e\n',relErrExactMSS2)
else
    relErrEulerRefEuler=meanRelError(XEulerRef(region1,region2,end,:),XEuler(region1,region2,end,:));
%     relErrEulerRefMSSC=meanRelError(XEulerRef(region1,region2,end,:),X3SSC(region1,region2,end,:));
%     relErrEulerRefMCS3=meanRelError(XEulerRef(region1,region2,end,:),X3CS(region1,region2,end,:));
    relErrEulerRefMCS2=meanRelError(XEulerRef(region1,region2,end,:),X2CS(region1,region2,end,:));
%     relErrEulerRefMSS3=meanRelError(XEulerRef(region1,region2,end,:),X3SS(region1,region2,end,:));
    relErrEulerRefMSS2=meanRelError(XEulerRef(region1,region2,end,:),X2SS(region1,region2,end,:));
    fprintf('Exact vs Euler : %3.3e\n',relErrEulerRefEuler)
%     fprintf('Exact vs SSC Magnus : %3.3e\n',relErrEulerRefMSSC)
%     fprintf('Exact vs iterated Magnus 3: %3.3e\n',relErrEulerRefMCS3)
    fprintf('Exact vs iterated Magnus 2: %3.3e\n',relErrEulerRefMCS2)
%     fprintf('Exact vs single Magnus 3: %3.3e\n',relErrEulerRefMSS3)
    fprintf('Exact vs single Magnus 2: %3.3e\n',relErrEulerRefMSS2)
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


