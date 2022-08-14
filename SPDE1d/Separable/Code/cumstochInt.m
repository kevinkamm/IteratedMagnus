function I=cumstochInt(f,dW)
    I=zeros(size(f,1),size(f,2),size(dW,3)+1,size(dW,4));
    if size(f,3)>1
        I(:,:,2:end,:)=cumsum(f(:,:,1:1:end-1,:).*dW,3); 
    else
        I(:,:,2:end,:)=f.*cumsum(dW,3); 
    end
end