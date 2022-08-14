function [errMatrix,errAverage]=absError(utRef,utApprox,kappas)
    [nx,nv] = size(utRef,[1,2]);
    errAverage=zeros(length(kappas),size(utRef,3));
    errMatrix=mean(abs(utRef-utApprox),4);
    for iKappa = 1:1:length(kappas)
        kappa=2^iKappa; 
        regionX=1+floor(nx/2-nx/(2*kappa)):1:ceil(nx/2+nx/(2*kappa));
        regionV=1+floor(nv/2-nv/(2*kappa)):1:ceil(nv/2+nv/(2*kappa));
        errAverage(iKappa,:)=mean(abs(utRef(regionX,regionV,:,:)-...
                                    utApprox(regionX,regionV,:,:)),[1,2,4]);
    end
end