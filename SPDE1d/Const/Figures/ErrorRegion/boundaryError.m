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

dT=.05;
dTSSC=2*dT;
tol=1e-2;

dtMagnusLog=10^(-4);
% dtEuler=10^(-3);
% dtEulerRef=10^(-4);


NMagnusLog= floor(T/dtMagnusLog)+1;
% NEulerRef= floor(T/dtEulerRef)+1;
% NEuler= floor(T/dtEuler)+1;

t=linspace(0,T,NMagnusLog);
[dWvec,tIndvec]=brownianIncrement(T,NMagnusLog,M);
dWMagnus=dWvec{1};
WMagnus=zeros(size(dWMagnus)+[0,0,1,0]);
WMagnus(1,1,2:end,:)=dWMagnus;
WMagnus=cumsum(WMagnus,3);

d=300;

method='HeatEquation1';

Ti=1:((NMagnusLog-1)./(T./dT)):NMagnusLog;
ticCoeff=tic;
[A,B,X0,XExact,region,EulerLangevin]=coefficients(t(Ti),M,d,WMagnus(1,1,Ti,:),method);
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
%% Errors
test=0;
kappas=4;
errAbsExactMagnus2={};
errAbsAvgExactMagnus2={};
errAbsExactMagnus3={};
errAbsAvgExactMagnus3={};
for i=2:length(Ti)
    [errMatrix,errAverage]=absError(XExact(:,:,i,:),X2CS(:,:,i,:),kappas);
    errAbsExactMagnus2{end+1}=errMatrix;
    errAbsAvgExactMagnus2{end+1}=errAverage;
    [errMatrix,errAverage]=absError(XExact(:,:,i,:),X3CS(:,:,i,:),kappas);
    errAbsExactMagnus3{end+1}=errMatrix;
    errAbsAvgExactMagnus3{end+1}=errAverage;
end
xigrid=linspace(-4,4,d+2);
xgrid=xigrid';
%%
figExactM2=plotAbsErrors(errAbsExactMagnus2,errAbsAvgExactMagnus2,xgrid(2:end-1),xigrid(2:end-1),kappas,t(Ti(2:end)));
figExactM3=plotAbsErrors(errAbsExactMagnus3,errAbsAvgExactMagnus3,xgrid(2:end-1),xigrid(2:end-1),kappas,t(Ti(2:end)));
%%
gifPath='Video/HeatEq';
fileName2=videoAbsErrors(gifPath,'ExactMagnus2',figExactM2);
fileName3=videoAbsErrors(gifPath,'ExactMagnus3',figExactM3);
disp('done')