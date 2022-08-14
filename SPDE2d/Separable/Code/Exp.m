function X=Exp(Y,Y0)
d=size(Y,1);
m=30;
precision = 'half';
switch size(Y0,2)
    case 0
        F=expm(Y);
    case 1
        [F,~,~,~]=expmvtay2(Y,Y0,m,precision);
    case d
        F=expm(Y)*Y0;
end
X=gather(F);
end