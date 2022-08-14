function fig=newFigure(varargin)
backgroundColor='w';
textColor = 'k';
figureRatio = 'fullScreen';
for iV=1:2:length(varargin)
    switch varargin{iV}
        case 'backgroundColor'
            backgroundColor = varargin{iV+1};
        case 'figureRatio'
            figureRatio = varargin{iV+1};
        case 'textColor'
            textColor  = varargin{iV+1};
    end
end
if strcmp(figureRatio,'square')
    fig=figure();
else
    fig=figure('units','normalized',...
               'outerposition',[0 0 1 1]);
end
hold on;
fig.Visible='off';
% fig.WindowState = 'minimized';
fontsize=22;
linewidth=2;
markersize=12;
set(gca,'FontSize',fontsize)
set(gca,'defaultLineMarkerSize',markersize)
set(fig,'defaultlinelinewidth',linewidth)
set(fig,'defaultaxeslinewidth',linewidth)
set(fig,'defaultpatchlinewidth',linewidth)
set(fig,'defaultAxesFontSize',fontsize)
set(gca, 'color', backgroundColor);
set(gca, 'XColor', textColor);
set(gca, 'YColor', textColor);
set(gca, 'ZColor', textColor);
end