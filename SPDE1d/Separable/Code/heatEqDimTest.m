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
dtMagnusLog=10^(-5);
dtEuler=10^(-3);
dtEulerRef=10^(-5);

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

for d=100:100:400
if d<=100
dT=.1;
elseif d<=200
dT=.025;
elseif d<=300
dT=.01;
else
dT=.005;
end




method='HeatEquation1';


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

% %% Single step
% ticMagnus2SS=tic;
% X2SS=magnusConst(A,B,X0,WMagnus,T,2);
% ctimeMagnus2SS=toc(ticMagnus2SS);
% fprintf('Elapsed time single step Magnus 2: %g s\n',ctimeMagnus2SS);
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)
% %%
% ticMagnus3SS=tic;
% X3SS=magnusConst(A,B,X0,WMagnus,T,3);
% ctimeMagnus3SS=toc(ticMagnus3SS);
% fprintf('Elapsed time single step Magnus 3: %g s\n',ctimeMagnus3SS);
% clearGPU=parfevalOnAll(@gpuDevice,0,[]);
% wait(clearGPU)

% %% Step size control
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
disp('Compute Euler reference')
if isempty(EulerLangevin)
    A=full(A);
    B=full(B);
end
ticEulerRef=tic;
XEulerRef=eulerSeparable(A,a,B,b,X0,dWEulerRef,T);
ctimeEulerRef=toc(ticEulerRef);
fprintf('Elapsed time EulerRef: %g s\n',ctimeEulerRef);
%%
disp('Compute Euler')
ticEuler=tic;
XEuler=eulerSeparable(A,aEuler,B,bEuler,X0,dWEuler,T);
ctimeEuler=toc(ticEuler);
fprintf('Elapsed time Euler: %g s\n',ctimeEuler)

%% Errors

relErrEulerRefEuler=meanRelError(XEulerRef(region1,region2,1,:),XEuler(region1,region2,1,:));
%     relErrEulerRefMSSC=meanRelError(XEulerRef(region1,region2,1,:),X3SSC(region1,region2,1,:));
relErrEulerRefMCS3=meanRelError(XEulerRef(region1,region2,1,:),X3CS(region1,region2,1,:));
relErrEulerRefMCS2=meanRelError(XEulerRef(region1,region2,1,:),X2CS(region1,region2,1,:));
% relErrEulerRefMSS3=meanRelError(XEulerRef(region1,region2,1,:),X3SS(region1,region2,1,:));
% relErrEulerRefMSS2=meanRelError(XEulerRef(region1,region2,1,:),X2SS(region1,region2,1,:));
fprintf('EulerRef vs Euler : %3.3e\n',relErrEulerRefEuler)
%     fprintf('EulerRef vs SSC Magnus : %3.3e\n',relErrEulerRefMSSC)
fprintf('EulerRef vs iterated Magnus 3: %3.3e\n',relErrEulerRefMCS3)
fprintf('EulerRef vs iterated Magnus 2: %3.3e\n',relErrEulerRefMCS2)
% fprintf('EulerRef vs single Magnus 3: %3.3e\n',relErrEulerRefMSS3)
% fprintf('EulerRef vs single Magnus 2: %3.3e\n',relErrEulerRefMSS2)

%% Save results
if strcmp(method,'Tridiag2x2exm1')
    fileName=[method,sprintf('_T%1.2f_d2_dtEulerRef%1.3e_dtLog%1.3e_dTiter%1.3e_M%d',T,dtEulerRef,dtMagnusLog,dT,M)];
else
    fileName=[method,sprintf('_T%1.2f_d%d_dtEulerRef%1.3e_dtLog%1.3e_dTiter%1.3e_M%d',T,d,dtEulerRef,dtMagnusLog,dT,M)];
end
matDir=['Results/Mat/DimTest','/',method];
outputMatlab(matDir,fileName,...
              T,dtEulerRef,dtEuler,dtMagnusLog,dT,M,region1,d,...
              ctimeEulerRef,ctimeEuler,ctimeMagnus2CS,ctimeMagnus3CS,...
              relErrEulerRefMCS2,relErrEulerRefMCS3,...
              relErrEulerRefEuler)
% matDir=['Results/Mat','/',method];
% if ~exist(matDir,'dir')
%     mkdir(matDir);
% end
% try
%     save([matDir,'/',fileName,'.mat'],'relErr*')
%     save([matDir,'/',fileName,'.mat'],'ctime*','-append')
% catch 
% end
end

