clear all; close all;
%% Parameter
a=1.1/10*10;
sigma=1/10*sqrt(10);
T=2;
M=1;
dt=1e-1;
N = floor(T/dt)+1;
%% grids
d=300;
xgrid=linspace(-4,4,d)';
xigrid=linspace(-4,4,d);
tgrid=linspace(0,T,N);
Ti=1:1:length(tgrid);
%% Brownian motion
W=zeros(1,1,N,M);
dW=sqrt(dt).*randn(1,1,N-1,M);
W(1,1,2:end,:)=cumsum(dW,3);
%% Solution of stoch Langevin with const coeff
utExact=exact(T,Ti,tgrid,xgrid,xigrid,a,sigma,W);
%% Video settings
backgroundColor='w';
textColor='k';
% backgroundColor=[53,54,58]./255;
% textColor=[237,237,237]./255;
fileDir=['Video','/',sprintf('T%1.3g_N%d_d%d',T,N,d)];
fileName='exact';
m=1;
%% File system
mkDir(fileDir);
file=[fileDir,'/',fileName];
delete([file,'.*']);
%% Make video
% auxiliary variables
[x,v]=meshgrid(xgrid,xigrid);

disp('Start to make the video')
h = waitbar(0,'Making video. Please wait...');
ticVideo=tic;
for ti = Ti
    ticFrame=tic;
    fig=newFigure('backgroundColor',backgroundColor,...
                  'textColor',textColor);
    s=surf(x,v,squeeze(utExact(:,:,ti,m))',...
           'FaceAlpha',.9,...
           'EdgeColor','none');
    view(3);
    zlim([0,1]);
    title(sprintf('t=%1.3f',tgrid(ti)),...
      'Interpreter','latex',...
      'Color',textColor);
%     exportgraphics(fig,[file,'.gif'],...
%                    'Append',true,...
%                    'BackgroundColor', backgroundColor);
    exportgraphics(fig,[file,'.pdf'],...
                   'Append',true,...
                   'BackgroundColor', backgroundColor);
    ctimeFrame=toc(ticFrame);
    waitbar(ti/length(Ti),h,...
            sprintf('One frame in %1.3f seconds',ctimeFrame));
end
ctimeVideo=toc(ticVideo);
fprintf('Elapsed time to make video: %1.3f seconds.\n',ctimeVideo);
delete(h);
close all;
%% Aux functions 
function mkDir(dir)
    if exist(dir)==0
        mkdir(dir);
    end
end