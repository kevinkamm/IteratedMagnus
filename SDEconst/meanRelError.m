function err=meanRelError(Xref,Xapprox)
    err=mean(vecnorm(vecnorm(Xref-Xapprox,2,1),2,2)./...
             vecnorm(vecnorm(Xref,2,1),2,2),4);
end