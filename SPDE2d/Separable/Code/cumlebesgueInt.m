function I=cumlebesgueInt(f,dt,T)
    I=zeros(size(f));
    if size(f,3)>1
        I(:,:,2:end,:)=cumsum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*reshape(linspace(0,T,round(T/dt)+1),1,1,[],1); 
    end
end