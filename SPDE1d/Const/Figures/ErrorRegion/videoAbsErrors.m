function fileName=videoAbsErrors(gifPath,gifName,absErrorRefFig,varargin)
backgroundColor='w';
for iV=1:2:length(varargin)
    switch varargin{iV}
        case 'backgroundColor'
            backgroundColor = varargin{iV+1};
    end
end
mkDir(gifPath);
fileName=[gifPath,'/',gifName];
delete([fileName,'.*']);
k=1;
for iFig = 1:1:length(absErrorRefFig)
    exportgraphics(absErrorRefFig{iFig},[fileName,'.gif'],'Append',true,'BackgroundColor', backgroundColor);
    exportgraphics(absErrorRefFig{iFig},[fileName,'.pdf'],...
                   'Append',true,...
                   'BackgroundColor', backgroundColor);
    k=k+1;
end
end
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