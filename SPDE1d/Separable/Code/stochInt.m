function I=stochInt(f,dW)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:).*dW,3); 
    else
        I=f.*sum(dW,3); 
    end
end