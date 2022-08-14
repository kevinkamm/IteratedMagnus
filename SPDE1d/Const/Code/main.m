clearvars; close all; fclose('all'); rng(0);
warning('off');
availGPU=gpuDeviceCount("available");
if availGPU > 0
    device='gpu';
    gpu=gpuDevice(1);
else
    device='cpu';
end
delete(gcp('nocreate'));parpool('threads'); % multithreading
%% Parameters
% Parameters for tri diag
% M=1000;
% T=1;
% 
% dT=.05;
% dTSSC=.05;
% tol=1e-3;
% dtMagnusLog=10^(-2);
% dtEuler=10^(-4);
% dtEulerRef=10^(-5);
% Parameters for Heat equation
M=100;
T=1;

dT=.005;
dTSSC=2*dT;
tol=1e-2;

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
WMagnus=zeros(size(dWMagnus)+[0,0,1,0]);
WMagnus(1,1,2:end,:)=dWMagnus;
WMagnus=cumsum(WMagnus,3);

d=500;
% method='Tridiag2x2exm1';
% method='HeatEquation1';
method='HeatEquationCauchy1';
% method='Langevin1';

ticCoeff=tic;
[A,B,X0,XExact,region1,region2,EulerLangevin]=coefficients(t,M,d,dWEulerRef,method);
ctimeCoeff=toc(ticCoeff);


%% Magnus
% % Constant step size
ticMagnus2CS=tic;
X2CS=magnusConstCS(A,B,X0,WMagnus,T,dT,2);
ctimeMagnus2CS=toc(ticMagnus2CS);
fprintf('Elapsed time iterated Magnus 2: %g s\n',ctimeMagnus2CS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)
%%     
ticMagnus3CS=tic;
X3CS=magnusConstCS(A,B,X0,WMagnus,T,dT,3);
ctimeMagnus3CS=toc(ticMagnus3CS);
fprintf('Elapsed time iterated Magnus 3: %g s\n',ctimeMagnus3CS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)

%% Single step
ticMagnus2SS=tic;
X2SS=magnusConst(A,B,X0,WMagnus,T,2);
ctimeMagnus2SS=toc(ticMagnus2SS);
fprintf('Elapsed time single step Magnus 2: %g s\n',ctimeMagnus2SS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)
%%
ticMagnus3SS=tic;
X3SS=magnusConst(A,B,X0,WMagnus,T,3);
ctimeMagnus3SS=toc(ticMagnus3SS);
fprintf('Elapsed time single step Magnus 3: %g s\n',ctimeMagnus3SS);
clearGPU=parfevalOnAll(@gpuDevice,0,[]);
wait(clearGPU)

%% Step size control
% ticMagnus3SSC=tic;
% [X3SSC,stepsizes]=magnusConstSSC(A,B,X0,WMagnus,T,dTSSC,tol,1:d);
% ctimeMagnus3SSC=toc(ticMagnus3SSC);
% fprintf('Elapsed time step size control Magnus 3: %g s\n',ctimeMagnus3SSC);
% if ~isempty(EulerLangevin)
%     X3SSC=reshape(X3SSC,d,d,1,M);
%     X3CS=reshape(X3CS,d,d,1,M);
%     X2CS=reshape(X2CS,d,d,1,M);
%     X3SS=reshape(X3SS,d,d,1,M);
%     X2SS=reshape(X2SS,d,d,1,M);
% end
%% Euler
if isempty(EulerLangevin)
    A=full(A);
    B=full(B);
end
ticEulerRef=tic;
if isempty(EulerLangevin)
    XEulerRef=eulerConst(A,B,X0,dWEulerRef,T);
else
    XEulerRef=eulerLangevin(T,dWEulerRef,EulerLangevin);
end
ctimeEulerRef=toc(ticEulerRef);
fprintf('Elapsed time EulerRef: %g s\n',ctimeEulerRef);

ticEuler=tic;
if isempty(EulerLangevin)
    XEuler=eulerConst(A,B,X0,dWEuler,dT);
else
    XEuler=eulerLangevin(T,dWEuler,EulerLangevin);
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
outputMatlab(matDir,fileName,...
              T,dtEulerRef,dtEuler,dtMagnusLog,dT,M,region1,d,...
              ctimeEulerRef,ctimeEuler,ctimeMagnus2CS,ctimeMagnus3CS,...
              relErrExactMCS2,relErrExactMCS3,...
              relErrExactEulerRef,relErrExactEuler)
% matDir=['Results/Mat','/',method];
% if ~exist(matDir,'dir')
%     mkdir(matDir);
% end
% try
%     save([matDir,'/',fileName,'.mat'],'relErr*')
%     save([matDir,'/',fileName,'.mat'],'ctime*','-append')
% catch 
% end


