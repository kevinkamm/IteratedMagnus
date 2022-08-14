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

dtMagnusLog=10^(-4);
dtEuler=[];
dtEulerRef=[];
relErrExactEulerRef=[];
relErrExactEuler=[];
ctimeEulerRef=[];
ctimeEuler=[];

NMagnusLog= floor(T/dtMagnusLog)+1;

t=linspace(0,T,NMagnusLog);
[dWvec,tIndvec]=brownianIncrement(T,NMagnusLog,M);
dWMagnus=dWvec{1};
WMagnus=zeros(size(dWMagnus)+[0,0,1,0]);
WMagnus(1,1,2:end,:)=dWMagnus;
WMagnus=cumsum(WMagnus,3);

d=200;

method='HeatEquation1';


ticCoeff=tic;
[A,B,X0,XExact,region,EulerLangevin]=coefficients(t,M,d,dWMagnus,method);
ctimeCoeff=toc(ticCoeff);
dTtemp=[5,2.5,1.25,1]'./(10.^(1:3));
for dT = unique(dTtemp(:)')
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
    
    %% Errors
    if ~isempty(XExact)
%         relErrExactEulerRef=meanRelError(XExact(region,region,1,:),XEulerRef(region,region,1,:));
%         relErrExactEuler=meanRelError(XExact(region,region,1,:),XEuler(region,region,1,:));
%         relErrExactMSSC=meanRelError(XExact(region,region,1,:),X3SSC(region,region,1,:));
        relErrExactMCS3=meanRelError(XExact(region,region,1,:),X3CS(region,region,1,:));
        relErrExactMCS2=meanRelError(XExact(region,region,1,:),X2CS(region,region,1,:));
%         relErrExactMSS3=meanRelError(XExact(region,region,1,:),X3SS(region,region,1,:));
%         relErrExactMSS2=meanRelError(XExact(region,region,1,:),X2SS(region,region,1,:));
%         fprintf('Exact vs EulerRef : %3.3e\n', relErrExactEulerRef)
%         fprintf('Exact vs Euler : %3.3e\n',relErrExactEuler)
%         fprintf('Exact vs SSC Magnus : %3.3e\n',relErrExactMSSC)
        fprintf('Exact vs iterated Magnus 3: %3.3e\n',relErrExactMCS3)
        fprintf('Exact vs iterated Magnus 2: %3.3e\n',relErrExactMCS2)
%         fprintf('Exact vs single Magnus 3: %3.3e\n',relErrExactMSS3)
%         fprintf('Exact vs single Magnus 2: %3.3e\n',relErrExactMSS2)
    else
%         relErrEulerRefEuler=meanRelError(XEulerRef(region,region,1,:),XEuler(region,region,1,:));
%         relErrEulerRefMSSC=meanRelError(XEulerRef(region,region,1,:),X3SSC(region,region,1,:));
%         relErrEulerRefMCS3=meanRelError(XEulerRef(region,region,1,:),X3CS(region,region,1,:));
%         relErrEulerRefMCS2=meanRelError(XEulerRef(region,region,1,:),X2CS(region,region,1,:));
%         relErrEulerRefMSS3=meanRelError(XEulerRef(region,region,1,:),X3SS(region,region,1,:));
%         relErrEulerRefMSS2=meanRelError(XEulerRef(region,region,1,:),X2SS(region,region,1,:));
%         fprintf('EulerRef vs Euler : %3.3e\n',relErrEulerRefEuler)
%         fprintf('EulerRef vs SSC Magnus : %3.3e\n',relErrEulerRefMSSC)
%         fprintf('EulerRef vs iterated Magnus 3: %3.3e\n',relErrEulerRefMCS3)
%         fprintf('EulerRef vs iterated Magnus 2: %3.3e\n',relErrEulerRefMCS2)
%         fprintf('EulerRef vs single Magnus 3: %3.3e\n',relErrEulerRefMSS3)
%         fprintf('EulerRef vs single Magnus 2: %3.3e\n',relErrEulerRefMSS2)
    end
    %% Save results
    if strcmp(method,'Tridiag2x2exm1')
        fileName=[method,sprintf('_T%1.2f_d2_dtLog%1.3e_dTiter%1.3e_M%d',T,dtMagnusLog,dT,M)];
    else
        fileName=[method,sprintf('_T%1.2f_d%d_dtLog%1.3e_dTiter%1.3e_M%d',T,d,dtMagnusLog,dT,M)];
    end
    matDir=['Results/Mat/StepTest','/',method];
    outputMatlab(matDir,fileName,...
                  T,dtEulerRef,dtEuler,dtMagnusLog,dT,M,region,d,...
                  ctimeEulerRef,ctimeEuler,ctimeMagnus2CS,ctimeMagnus3CS,...
                  relErrExactMCS2,relErrExactMCS3,...
                  relErrExactEulerRef,relErrExactEuler)
end
% matDir=['Results/Mat','/',method];
% if ~exist(matDir,'dir')
%     mkdir(matDir);
% end
% try
%     save([matDir,'/',fileName,'.mat'],'relErr*')
%     save([matDir,'/',fileName,'.mat'],'ctime*','-append')
% catch 
% end


