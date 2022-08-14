function [A,B,X0,Xexact,region1,region2]=coefficients(t,M,d,dW1,example)
    N=length(t);
    t=reshape(t,1,1,[],1);
    T=t(end);
    dt=T/(N-1);
%     bdFunc=@(x) 1./(1+exp(-x));%sigmoid
%     bdFunc=@(x) 5.*exp(-x.^2);%Gauss
%     bdFunc=@(x) 2.*(1/(1+x.^2));
%     bdFunc=@(x) 2.*atan(x);
%     bdFunct=@(t,x) 2.*atan(x).*t;

    bdFunc=@(x) (1+1/(1+x.^2));
    bdFunctB=@(t,x) bdFunc(x);
    bdFunctA=@(t,x) sqrt(bdFunc(x));
    switch example
        case 'Tridiag2x2exm1'
            dW2=sqrt(dt).*randn(1,1,N-1,M,6);
            W2=zeros(1,1,N,M,6);
            W2(1,1,2:end,:,:)=dW2;
            W2=cumsum(W2,3);
%             a=rand(2,2);
%             b=rand(2,2);
%             a(2,1)=0;
%             b(2,1)=0;
%             disp(a);
%             disp(b);
            a=[0.089169311678098,0.401498312870165;0,0.016873613318980];
            b=[0.923389293105627,0.381331107923466;0,0.540312021308851];
            disp(a);
            disp(b);
            A=zeros(2,2,N,M);
            B=zeros(2,2,N,M);
            A(1,1,:,:)=a(1,1).*bdFunctA(t,W2(1,1,:,:,1));
            A(1,2,:,:)=a(1,2).*bdFunctA(t,W2(1,1,:,:,2));
            A(2,2,:,:)=a(2,2).*bdFunctA(t,W2(1,1,:,:,3));
            B(1,1,:,:)=b(1,1).*bdFunctB(t,W2(1,1,:,:,4));
            B(1,2,:,:)=b(1,2).*bdFunctB(t,W2(1,1,:,:,5));
            B(2,2,:,:)=b(2,2).*bdFunctB(t,W2(1,1,:,:,6));
            X0=[];
            Xexact=exactTriDiag2x2(X0,A,B);
           region1=1;
           region2=2;
        otherwise
            error('Unknown example')

    end
    function X=exactTriDiag2x2(X0,A,B)
        if isempty(X0)
            X0=[1 0; 0 1];
        end
        X=zeros(2,2,1,M);
        Ia2ds=cumlebesgueInt(A.^2,dt,T);
        Ibds=cumlebesgueInt(B,dt,T);
        IadWs=cumstochInt(A,dW1);
        X(1,1,:,:)=X0(1,1).*exp(Ibds(1,1,end,:)+...
                                IadWs(1,1,end,:)-...
                                Ia2ds(1,1,end,:)./2);
        X22temp=X0(2,2).*exp(Ibds(2,2,:,:)+...
                                IadWs(2,2,:,:)-...
                                Ia2ds(2,2,:,:)./2);
        X(2,2,:,:)=X22temp(1,1,end,:);

        EZ=exp(Ibds(1,1,end,:)+...
                IadWs(1,1,end,:)-...
                Ia2ds(1,1,end,:)./2);
        %exp(-Z +<Z>/2)
        EZinv=exp(-(Ibds(1,1,:,:)+IadWs(1,1,:,:)-Ia2ds(1,1,:,:)./2));
        dHLeb=B(1,2,:,:).*X22temp;
        dHStoch=A(1,2,:,:).*X22temp;
        dHZ=-dHStoch.*A(1,1,:,:);
        X(1,2,:,:)=EZ.*(X0(1,2)+...
                    lebesgueInt(EZinv.*(dHLeb+dHZ),dt,T)+...
                    stochInt(EZinv.*dHStoch,dW1));
        
    end
end
function I=lebesgueInt(f,dt,T)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*T; 
    end
end
function I=stochInt(f,dW)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:).*dW,3); 
    else
        I=f.*sum(dW,3); 
    end
end
function I=cumlebesgueInt(f,dt,T)
    I=zeros(size(f));
    if size(f,3)>1
        I(:,:,2:end,:)=cumsum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*reshape(linspace(0,T,round(T/dt)+1),1,1,[],1); 
    end
end
function I=cumstochInt(f,dW)
    I=zeros(size(f,1),size(f,2),size(dW,3)+1,size(dW,4));
    if size(f,3)>1
        I(:,:,2:end,:)=cumsum(f(:,:,1:1:end-1,:).*dW,3); 
    else
        I(:,:,2:end,:)=f.*cumsum(dW,3); 
    end
end