clear all; close all;
tempPath='Data';
figPath=pwd;
fileDir=[pwd,'/',tempPath,'/*.mat'];
files=dir(fileDir);

% backgroundColor = [53,54,58]./255;
% textColor = [237,237,237]./255;
backgroundColor = 'None';
textColor='k';

ref='EulerRef';
errType='relError';

saveDir=[figPath,'/',errType];
mkDir(saveDir);

excludeMethods={'EulerRef'};


ctimes=zeros(length(files),3);
errors=zeros(length(files),3);
stepDt=zeros(length(files),3);

for iF = 1:1:length(files)
    file=[files(iF).folder,'/',files(iF).name];
    curr=load(file);
    tempMethods=fieldnames(curr.Result);
    for iM = 1:1:length(tempMethods)
        method=tempMethods{iM};
        if any(contains(excludeMethods,method),'all')
            continue;
        end   
        ctimes(iF,str2num(method(end)))=curr.Result.(method).ctime;
        errors(iF,str2num(method(end)))=curr.Result.(method).(ref).(errType)(end);
        stepDt(iF,str2num(method(end)))= curr.Result.(method).dt;
    end
end
M=curr.Result.Magnus3.M;
dtLog=curr.Result.Magnus3.dtLog;
d=curr.Result.Magnus3.d;
T=curr.Result.Magnus3.T;

[stepDtSorted,I]=sort(stepDt(:,end),1);
ctimesSorted=ctimes(I,:)./M;
errorsSorted=errors(I,:);
errorsSorted(isnan(errorsSorted))=1e2;

iDtMax=find(stepDtSorted<=0.05,1,'last');
stepDtSorted=stepDtSorted(1:iDtMax,:);
ctimesSorted=ctimesSorted(1:iDtMax,:);
errorsSorted=errorsSorted(1:iDtMax,:);

legendEntry={};
fig=newFigure('visible','on',...
              'backgroundColor',backgroundColor,...
              'textColor',textColor);hold on;

% title(sprintf('T=%1.2f, M=%d, d=%d, dtLog=%1.2e',T,M,d,dtLog),...
%         'Interpreter','latex',...
%         'Color',textColor,...
%         'FontSize',16);
xlabel('$\Delta_t$','Interpreter','latex')

yyaxis left; 
plot(stepDtSorted,ctimesSorted(:,2),'Color',[0.3010 0.7450 0.9330],'LineStyle',':')
legendEntry{end+1}='ctime m2';
plot(stepDtSorted,ctimesSorted(:,3),'Color',[0 0.4470 0.7410],'LineStyle','--')
legendEntry{end+1}='ctime m3';
ylabel('Avg. Comp. Time in sec. per simulation (log scale)')
set(gca, 'YScale', 'log')

yyaxis right; 
if strcmp(errType,'relError')
    plot(stepDtSorted,errorsSorted(:,2).*100,'Color',[0.8500 0.3250 0.0980],'LineStyle',':')
    legendEntry{end+1}='error m2';
    plot(stepDtSorted,errorsSorted(:,3).*100,'Color',[0.6350 0.0780 0.1840],'LineStyle','--')
    legendEntry{end+1}='error m3';
    ylabel('Error in % (log scale)')
    set(gca, 'YScale', 'log')
%     m=median(errorsSorted.*100);
%     m=m(3);
%     ylim([m.*.95,m.*10])
    ylim([0.009,0.1])
    a = [cellstr(num2str(get(gca,'ytick')'))];
    pct = char(ones(size(a,1),1)*'%');
    new_yticks = [char(a),pct]; 
    set(gca,'yticklabel',new_yticks) 
else
    plot(stepDtSorted,errorsSorted(:,2),'Color',[0.8500 0.3250 0.0980],'LineStyle',':')
    legendEntry{end+1}='error m2';
    plot(stepDtSorted,errorsSorted(:,3),'Color',[0.6350 0.0780 0.1840],'LineStyle','--')
    legendEntry{end+1}='error m3';
    ylabel('Error (log scale)')
    set(gca, 'YScale', 'log')
    m=median(errorsSorted);
    m=m(3);
    ylim([m/10,m*10])
end

legend(legendEntry,'Location','southoutside','NumColumns',4,'Color',backgroundColor)
exportgraphics(fig,[saveDir,'/',sprintf('StepTest_T%1.2f_M%d_d%d_dtLog%1.2e.pdf',T,M,d,dtLog)],...
                       "BackgroundColor",backgroundColor);
function delDir(dir)
    if exist(dir)==7
        rmdir(dir,'s');
    end
end
function mkDir(dir)
    if exist(dir)==0
        mkdir(dir);
    end
end
function delFile(file)
    if exist(file)
        delete(file);
    end
end
% dark blue: [0 0.4470 0.7410]
% orange: [0.8500 0.3250 0.0980]
% yellow: [0.9290 0.6940 0.1250]
% purple: [0.4940 0.1840 0.5560]
% green: [0.4660 0.6740 0.1880]
% light blue: [0.3010 0.7450 0.9330]
% red: [0.6350 0.0780 0.1840]