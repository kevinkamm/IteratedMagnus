function outputMatlab(matRoot,fileName,...
                      T,dtEulerRef,dtEuler,dtMagnusLog,dT,M,region,d,...
                      ctimeEulerRef,ctimeEuler,ctimeMagnus2CS,ctimeMagnus3CS,...
                      relErrExactMCS2,relErrExactMCS3,...
                      relErrExactEulerRef,relErrExactEuler)

mkDir(matRoot)
Result=struct;
kappa=round(log2(d/length(region)));

if ~isempty(dtEulerRef)
Result.EulerRef=struct;
Result.EulerRef=addMethod(Result.EulerRef,ctimeEulerRef,dtEulerRef);
Result.EulerRef=addRelError(Result.EulerRef,'Exact',relErrExactEulerRef);
end
if ~isempty(dtEuler)
Result.Euler=struct;
Result.Euler=addMethod(Result.Euler,ctimeEuler,dtEuler);
Result.Euler=addRelError(Result.Euler,'Exact',relErrExactEuler);
end

if ~isempty(relErrExactMCS2)
Result.Magnus2=struct;
Result.Magnus2=addMethod(Result.Magnus2,ctimeMagnus2CS,dT,dtMagnusLog);
Result.Magnus2=addRelError(Result.Magnus2,'Exact',relErrExactMCS2);
end

if ~isempty(relErrExactMCS3)
Result.Magnus3=struct;
Result.Magnus3=addMethod(Result.Magnus3,ctimeMagnus3CS,dT,dtMagnusLog);
Result.Magnus3=addRelError(Result.Magnus3,'Exact',relErrExactMCS3);
end

save([matRoot,'/',fileName,'.mat'],'Result')
    function sct=addMethod(sct,ctime,dt,dtLog)
        if nargin>3
            sct.dtLog=dtLog;
        end
        sct.ctime=ctime;
        sct.dt=dt;
        sct.kappas=kappa;
        sct.d=d;
        sct.T=T;
        sct.M=M;
    end
    function sct=addRelError(sct,ref,relError)
        sct.(ref).relError=relError;
    end
end
function delFile(file)
    if exist(file)
        delete(file);
    end
end
function delDir(dir)
    if exist(dir)==7
        rmdir(dir,'s');
    end
end
function cleanDir(mdir,except)
    except{end+1}='.';
    except{end+1}='..';
    for d = dir(mdir).'
      if ~any(strcmp(d.name,except))
          if d.isdir
              rmdir([d.folder,'/',d.name],'s');
          else
              delete([d.folder,'/',d.name]);
          end
      end
    end
end
function mkDir(dir)
    if exist(dir)==0
        mkdir(dir);
    end
end