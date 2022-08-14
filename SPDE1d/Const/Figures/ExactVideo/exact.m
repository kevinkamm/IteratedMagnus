function utExact=exact(T,Ti,timegrid,xgrid,xigrid,a,sigma,W)
%     N=size(W,3);
%     M=size(W,4);
    
%     dt=T/(N-1);
%     IW=l_int(W,dt);
    
    x=xgrid;
    xi=xigrid;
    t=reshape(timegrid(Ti),1,1,[],1);
    Wt=W(1,1,Ti,:);
    utExact=...
        exp(-(x+sigma.*Wt-xi).^2./(2.*(a-sigma^2).*t))./...
        sqrt(2.*pi.*(a-sigma^2).*t);
    if Ti(1)==1 && timegrid(1)==0
        utExact(:,:,1,:)=eye(length(x));
    end
end
