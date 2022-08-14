clear all; close all;
figPath=pwd;
fileDir=[pwd,'/','Data','/*.mat'];
files=dir(fileDir);

% backgroundColor = [53,54,58]./255;
% textColor = [237,237,237]./255;
backgroundColor = 'w';
textColor='k';
% coefficients='exact1';
% ref='Exact';
% excludeLogDts=[1e-2];
coefficients='Langevin1';
ref='EulerRef';
excludeLogDts=[];

% errType='absError';
errType='relError';

saveDir=[figPath,'/',errType];
mkDir(saveDir);

excludeMethods={'Magnus1'};

excludeEulerDt=[1e-2,1e-5];

tempDims = zeros(length(files),1);

for iF = 1:1:length(files)
    if ~contains(files(iF).name,coefficients)
        continue;
    end
    file=[files(iF).folder,'/',files(iF).name];
    curr=load(file);
    tempMethods=fieldnames(curr.Result);
    method=tempMethods{1};
    tempDims(iF) = curr.Result.(method).d;
end
uniqueDims = unique(tempDims(tempDims>0));

for iD = 1:1:length(uniqueDims)
    dim = uniqueDims(iD);
    indFiles = tempDims == dim;
    currFiles = files(indFiles);
    dimPlot(currFiles,excludeMethods,excludeLogDts,excludeEulerDt,ref,errType,coefficients,saveDir,backgroundColor,textColor);
end

function dimPlot(files,excludeMethods,excludeLogDts,excludeEulerDt,ref,errType,coefficients,saveDir,backgroundColor,textColor)
    
    
    ctimes=[];
    dims=[];
    dts=[];
    errors=[];
    methods={};
    methodsStr={};
    orders=[];
    logDts=[];
    M=[];
    for iF = 1:1:length(files)
        file=[files(iF).folder,'/',files(iF).name];
        curr=load(file);
        tempMethods=fieldnames(curr.Result);
        for iM = 1:1:length(tempMethods)
            method=tempMethods{iM};
            if any(contains(excludeMethods,method),'all')
                continue;
            end
            if ~isfield(curr.Result.(method),ref)
                curr.Result.(method).ctime
                continue;
            end
            currError = curr.Result.(method).(ref).(errType)(end);
            currCtime = curr.Result.(method).ctime;
            currDt = curr.Result.(method).dt;
            currM = curr.Result.(method).M;
            currD = curr.Result.(method).d;
            currMethod = replace(method,'Ref','');
            currMethodStr = [currMethod,sprintf(', dt=%1.1e, d=%d',currDt,currD)];
            currOrder = str2num(currMethod(end));
            if ~isempty(currOrder)
%                 currMethod=currMethod(1:end-1);
                currMethod = [currMethod(1),currMethod(end),sprintf(', %1.1e, d=%d',currDt,currD)];
            else
                currOrder=-1;
                currMethod = [currMethod(1),sprintf(', %1.1e, d=%d',currDt,currD)];
            end
%             currMethod = [currMethod(1),sprintf(', %1.1e, d=%d',currDt,currD)];
            currLogDt=-1;
            if strcmp(currMethod(1),'M')
                currLogDt = curr.Result.(method).dtLog;
%                 currMethod = [currMethod,sprintf(', %1.1e',currLogDt)];
                currMethodStr = [currMethodStr,sprintf(', %1.1e',currLogDt)];
            elseif strcmp(currMethod(1),'E') && any(currDt == excludeEulerDt)
                continue;
            end
            if any(currLogDt==excludeLogDts)
                continue;
            end
            ctimes(end+1)=currCtime;
            errors(end+1)=currError;
            methods{end+1}=currMethod;
            methodsStr{end+1}=currMethodStr;
            dims(end+1)=currD;
            dts(end+1)=currDt;
            orders(end+1)=currOrder;
            logDts(end+1)=currLogDt;
            M(end+1)=currM;
        end
    end
    
    currMethods=methods;
    currMethodsStr=methodsStr;
    currCtimes=ctimes;
    currErrors=errors;
    currOrders=orders;
    currDims=dims;
    currDts=dts;
    currLogDts=logDts;
    M=unique(M);
    if length(M)>1
        error('Corrupt data set, different simulation sizes. Check M!')
    end
    fig=newFigure('visible','on',...
                  'backgroundColor',backgroundColor,...
                  'textColor',textColor);hold on;
    
    [uniqueMethodsStr,iU,~]=unique(currMethodsStr);
    x = currMethods(iU);
    c = reshape(currCtimes(iU),[],1)./M;
    if strcmp(errType,'relError')
        e = reshape(currErrors(iU),[],1).*100;
    else
        e = reshape(currErrors(iU),[],1);
    end
    o = currOrders(iU);
    dt = currDts(iU);
    d = currDims(iU);
    logDt = currLogDts(iU);
%     [x,c,e,o,xD,yD,xEuler,xMagnus2,xMagnus3,yEuler,yMagnus2,yMagnus3]=sortMethods(d,x,c,e,o,dt,logDt);
    t = tiledlayout(1,1,'TileSpacing','none');
%         t = tiledlayout('flow','TileSpacing','none');
%     bgAx = axes(t,'XTick',[],'YTick',[],'Box','off');
%         bgAx.Layout.TileSpan = [1 length(x)];
%     ax = {};

        hold off;
%     xEuler = o == -1 & e' < 1e6;
%     xMagnus2 = o == 2 & e' < 1e6;
%     xMagnus3 = o == 3 & e' < 1e6;
    e(e>1e2 | isnan(e))=inf;
    xEuler = o == -1;
    xMagnus2 = o == 2;
    xMagnus3 = o == 3;
    currEX = categorical(x(xEuler),string(x(xEuler)));
    currM3X = categorical(x(xMagnus3),string(x(xMagnus3)));
    currM2X = categorical(x(xMagnus2),string(x(xMagnus2)));
%     currX=[currEX,currM3X,currM2X];
%             ax{end+1} = axes(t);
        currAx = nexttile;hold on;
        set(gca, 'color', backgroundColor);
        set(gca, 'XColor', textColor);
%             set(gca,'FontSize',22)
        
        yyaxis left
        % ctimes euler
        if length(currEX)>0
            bEuler=ctimeBars(currEX,c(xEuler),[],textColor,[0 0.4470 0.7410],'bottom');
        end
        % ctimes Magnus 2
        if length(currM2X)>0
%             bMagnus2=ctimeBars(currM2X,c(xMagnus2),.25,textColor,[0.3010 0.7450 0.9330],'top');
            bMagnus2=ctimeBars(currM2X,c(xMagnus2),[],textColor,[0 0.4470 0.7410],'bottom');
        end
        % ctimes Magnus 3
        if length(currM3X)>0
            bMagnus3=ctimeBars(currM3X,c(xMagnus3),[],textColor,[0 0.4470 0.7410],'bottom');
        end
        set(gca, 'YScale', 'log')
        tempYlim=ylim;
        tempYlim=tempYlim + tempYlim/3 .* [-1 1];
        ylim(tempYlim)
        ylabel('Avg. Comp. Time in sec. per simulation (log scale)')

        yyaxis right
        % errors euler
        if length(currEX)>0
            bEuler=errorBars(currEX,e(xEuler),[],textColor,[0.6350 0.0780 0.1840],'bottom',errType);
        end
        % errors Magnus 2
        if length(currM2X)>0
            bMagnus2=errorBars(currM2X,e(xMagnus2),[],textColor,[0.6350 0.0780 0.1840],'bottom',errType);
%             bMagnus2=errorBars(currM2X,e(xMagnus2),.25,textColor,[0.8500 0.3250 0.0980],'top');
        end
        % errors Magnus 3
        if length(currM3X)>0
            bMagnus3=errorBars(currM3X,e(xMagnus3),[],textColor,[0.6350 0.0780 0.1840],'bottom',errType);
        end
%         tempYlim=ylim;
%         tempYlim=tempYlim + tempYlim/5 .* [-1 1];
%         ylim(tempYlim)
%         set(gca, 'YScale', 'log')
        if strcmp(errType,'relError')
            ylabel('Error in % (linear scale)')
            a = [cellstr(num2str(get(gca,'ytick')'))];
            pct = char(ones(size(a,1),1)*'%');
            new_yticks = [char(a),pct]; 
            set(gca,'yticklabel',new_yticks) 
        else
            ylabel('Error (linear scale)')
        end

        XTickLabelRotation=90;
        currXTickLabels=xticklabels;
        xticklabels(replace(replace(currXTickLabels,[', d='+digitsPattern],' '),'*',''));
%             currAx.XAxis.TickLabelInterpreter='latex';
%             currAx.XAxis.TickLabelColor=textColor;
%             xlabel(replace(replace(currXTickLabels,[', d='+digitsPattern],' '),'*','\\newline'),"Interpreter","latex")
        currAx.XAxis.Color=textColor;
        currAx.XAxis.LineWidth=3;



    title(sprintf('Space grid with $d=%d$ points',...
            d(1)),...
            'Interpreter','latex',...
            'Color',textColor,...
            'FontSize',22);
    exportgraphics(fig,[saveDir,'/',coefficients,'_',sprintf('dim_%d.pdf',d(1))],...
                   "BackgroundColor",backgroundColor);
end
function b=ctimeBars(x,c,w,textColor,faceColor,vAlignment)
    if isempty(w)
        b=bar(x,[c,nan.*ones(length(x),1)],'FaceColor',faceColor);
    else
        b=bar(x,[c,nan.*ones(length(x),1)],w,'FaceColor',faceColor);
    end
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = strsplit(sprintf("%1.3f ",b(1).YData),' ');
    text(xtips1,ytips1,labels1(1:end-1),'HorizontalAlignment','center',...
        'VerticalAlignment',vAlignment,'Color',textColor,'FontSize',16)
end
function b=errorBars(x,e,w,textColor,faceColor,vAlignment,errType)
    indInf = isinf(e);
    if ~isempty(e(~indInf))
        e(indInf)=max(e(~indInf)).*1.5;
    else
        e(indInf)=1;
    end
    unit='';
    if strcmp(errType,'relError')
        unit=' %';
    end
    if isempty(w)
        b=bar(x,[nan.*ones(length(x),1),e],'FaceColor',faceColor);
    else
        b=bar(x,[nan.*ones(length(x),1),e],w,'FaceColor',faceColor);
    end
    xtips2 = b(2).XEndPoints;
    ytips2 = b(2).YEndPoints;
    ydata=b(2).YData(:);

    ydata(indInf)=inf;
%     b(2).YData(indInf)=inf;
%     indInf = isinf(e);
%     ytips2(indInf)=0;

    labels2 = strsplit(sprintf("%1.3f ",ydata),' ');
    labels2 = arrayfun(@(x,s) [x+unit],labels2);
    text(xtips2,ytips2,labels2(1:end-1),'HorizontalAlignment','center',...
        'VerticalAlignment',vAlignment,'Color',textColor,'FontSize',16)
end

% dark blue: [0 0.4470 0.7410]
% orange: [0.8500 0.3250 0.0980]
% yellow: [0.9290 0.6940 0.1250]
% purple: [0.4940 0.1840 0.5560]
% green: [0.4660 0.6740 0.1880]
% light blue: [0.3010 0.7450 0.9330]
% red: [0.6350 0.0780 0.1840]
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