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

% dT=.05;
% dTSSC=2*dT;
tol=1e-2;

relErrEulerRefEuler=[];
ctimeEuler=[];

dtMagnusLog=10^(-4);
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
dtEuler=[];
d=200;

method='HeatEquation1';


ticCoeff=tic;
[A,a,B,b,X0,XExact,region1,region2,EulerLangevin]=coefficients(t,M,d,dWMagnus,method);
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

%% Euler
disp('Compute Euler reference')
ticEulerRef=tic;
XEulerRef=eulerSeparable(full(A),a,full(B),b,X0,dWEulerRef,T);
ctimeEulerRef=toc(ticEulerRef);
fprintf('Elapsed time EulerRef: %g s\n',ctimeEulerRef);

dTtemp=[5,2.5,1.25,1]'./(10.^(1:3));
for dT = flip(unique(dTtemp(:)'))
    %% Magnus
    % % Constant step size
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
    
    
    %% Errors

        relErrEulerRefMCS3=meanRelError(XEulerRef(region1,region2,1,:),X3CS(region1,region2,1,:));
        relErrEulerRefMCS2=meanRelError(XEulerRef(region1,region2,1,:),X2CS(region1,region2,1,:));
        fprintf('EulerRef vs iterated Magnus 3: %3.3e\n',relErrEulerRefMCS3)
        fprintf('EulerRef vs iterated Magnus 2: %3.3e\n',relErrEulerRefMCS2)
    %% Save results
    if strcmp(method,'Tridiag2x2exm1')
        fileName=[method,sprintf('_T%1.2f_d2_dtLog%1.3e_dTiter%1.3e_M%d',T,dtMagnusLog,dT,M)];
    else
        fileName=[method,sprintf('_T%1.2f_d%d_dtLog%1.3e_dTiter%1.3e_M%d',T,d,dtMagnusLog,dT,M)];
    end
    matDir=['Results/Mat/StepTestSep','/',method];
    outputMatlab(matDir,fileName,...
                  T,dtEulerRef,dtEuler,dtMagnusLog,dT,M,region1,d,...
                  ctimeEulerRef,ctimeEuler,ctimeMagnus2CS,ctimeMagnus3CS,...
                  relErrEulerRefMCS2,relErrEulerRefMCS3,...
                  relErrEulerRefEuler)
end


