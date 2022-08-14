function figures=plotAbsErrors(errAbsRefApprox,errAbsAvgRefApprox,xgrid,vgrid,kappas,tEval,varargin)
%%PLOTABSERRORS plots absolute errors in a 2d matrix. Each entry is
% represented by a square with different color corresponding to the size of
% the error. 
%   Input:
%   Output:
%   Usage:
%
% See also exact, magnusConst, eulerConst.
showTitle=true;
showLines=true;
fixScale=true;
backgroundColor='w';
textColor='k';
for iV=1:2:length(varargin)
    switch varargin{iV}
        case 'showTitle'
            showTitle=varargin{iV+1};
        case 'showLines'
            showLines=varargin{iV+1};
        case 'fixScale'
            fixScale=varargin{iV+1};
        case 'backgroundColor'
            backgroundColor = varargin{iV+1};
        case 'textColor'
            textColor = varargin{iV+1};
    end
end

kappas=kappas(1);

figures={};
nx=length(xgrid);
nv=length(vgrid);

[exact,eulerRef,euler,m1,m2,m3,m4,methodDict]=methodColors();
methods=strsplit(regexprep(replace(inputname(1),'errAbs',''), '([A-Z])', ' $1'),' ');
i=2;
tempName= methods{2};
if strcmp(methods{2},'Euler')
if strcmp(methods{3},'Ref')
    tempName=[tempName,methods{3}];
    i=i+1;
end
end
refName=methodDict.(tempName);
i=i+1;
tempName= methods{i};
if strcmp(methods{i},'Euler')
if (i<length(methods)) && (strcmp(methods{i+1},'Ref'))
    tempName=[tempName,methods{i+1}];
end
end
approxName=methodDict.(tempName);
% refName=methodDict.(methods{2});
% approxName=methodDict.(methods{3});
if fixScale
    temp=[errAbsRefApprox{:}];
    cmin=min(temp,[],'all');
    cmax=max(temp,[],'all');
end
%% Figure 1
for kk=1:1:length(errAbsRefApprox)
    fig=newFigure('backgroundColor',backgroundColor,'textColor',textColor);
    [plots,legendEntries]=beginFigure();
    plot_AbsErrors(kk)
    if showLines
        plot_AvgErrors(kk)
    end
    endFigure(plots,legendEntries,'x','y',...
        sprintf('Mean Abs Error %s vs %s at t=%1.3g',refName,approxName,tEval(kk)));
end

%% Plot functions
function plot_AbsErrors(kk)
    [x,v]=meshgrid(xgrid,vgrid);
    s=surf(x,v,errAbsRefApprox{kk}','FaceAlpha',.9,'EdgeColor','none');hold on;
    plots(end+1)=s;
    view(2);
    colorbar('Color',textColor);
    colorbar('Ticks',[1e-5*10.^(0:4)],...
             'TickLabels',{'10^{-5}','10^{-4}','10^{-3}','10^{-2}','10^{-1}'},...
             'Limits',[1e-5,1e-1]);
    clim([1e-5,1e-1]);
%     if fixScale
%         clim([cmin/10,cmax*10]);
%     end
    set(gca,'ColorScale','log') 
    legendEntries{end+1}=sprintf('Mean Abs Error %s vs %s',refName,approxName);
end
function plot_AvgErrors(kk)
    for iKappa = 1:1:length(kappas)
        kappa=2^iKappa; 
        nz=floor(nx/(4*2*kappa));
        regionX=[1+floor(nx/2-nx/(2*kappa)),ceil(nx/2+nx/(2*kappa))];
        regionV=[1+floor(nv/2-nv/(2*kappa)),ceil(nv/2+nv/(2*kappa))];
        %lower left corner
        llCorner = [xgrid(regionX(1)),vgrid(regionV(1))];
        %upper left corner
        ulCorner = [xgrid(regionX(1)),vgrid(regionV(2))];
        %lower left corner
        lrCorner = [xgrid(regionX(2)),vgrid(regionV(1))];
        %upper right corner
        urCorner = [xgrid(regionX(2)),vgrid(regionV(2))];
        topLine(linspace(llCorner(1),ulCorner(1),nz),...
                linspace(llCorner(2),ulCorner(2),nz))
        topLine(linspace(lrCorner(1),urCorner(1),nz),...
                linspace(lrCorner(2),urCorner(2),nz))
        topLine(linspace(llCorner(1),lrCorner(1),nz),...
                linspace(llCorner(2),lrCorner(2),nz))
        topLine(linspace(ulCorner(1),urCorner(1),nz),...
                linspace(ulCorner(2),urCorner(2),nz))
        if length(kappas)>1
        text((llCorner(1)+lrCorner(1))./2,lrCorner(2),2,sprintf('%1.3e',errAbsAvgRefApprox{kk}(iKappa)),...
            'VerticalAlignment','bottom','HorizontalAlignment','center',...
            'FontSize',16,'fontweight', 'bold')
        else
        text((llCorner(1)+lrCorner(1))./2,lrCorner(2),2,sprintf('%1.3e',errAbsAvgRefApprox{kk}(iKappa)),...
            'VerticalAlignment','bottom','HorizontalAlignment','center',...
            'FontSize',24,'fontweight', 'bold')
        end
    end
end

%% Auxiliary functions
function topLine(x,y)
    x=x(:);
    y=y(:);
    patch([x;NaN],[y;NaN],2*ones(length(x)+1,1),'w--','linewidth',1)
    plot(x,y,'k--','linewidth',1)
end
function [plots,legendEntries]=beginFigure()
    figures{end+1}=fig;
    legendEntries = {};
    plots = [];
end
function endFigure(plots,legendEntries,xLabel,yLabel,titleStr)
    legend(plots,legendEntries,...
      'Location','southoutside',...
      'NumColumns',2,...
      'Interpreter','latex',...
      'TextColor',textColor); 
    if ~strcmp(xLabel,'')
        xlabel(xLabel, 'fontweight', 'bold')
    end
    if ~strcmp(yLabel,'')
        ylabel(yLabel, 'fontweight', 'bold')
    end
    if ~strcmp(titleStr,'') && showTitle
        title(titleStr,'Color',textColor,'Interpreter','latex')
    end
end
end