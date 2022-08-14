function I=lebesgueInt(f,dt,T)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*T; 
    end
end