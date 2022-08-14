function [A,B,X0,Xexact,region,EulerLangevin]=coefficients(t,M,d,W1,example)
%     N=length(dW1,3)+1;
    t=reshape(t,1,1,[],1);
%     T=t(end);
%     dt=T/(N-1);
%     W1=zeros(size(dW1)+[0 0 1 0]);
%     W1(1,1,2:end,:)=dW1;
%     W1=cumsum(W1,3);
%     bdFunc=@(x) 1./(1+exp(-x));%sigmoid
%     bdFunc=@(x) 5.*exp(-x.^2);%Gauss
%     bdFunc=@(x) 2.*(1/(1+x.^2));
%     bdFunc=@(x) 2.*atan(x);
%     bdFunct=@(t,x) 2.*atan(x).*t;
    kappa=4;
    EulerLangevin=[];
    switch example
        case 'HeatEquation1'
            x=linspace(-4,4,d+2)';
            dx=(x(end)-x(1))./(length(x)-1);
            a=1.1;
            sigma=1/sqrt(10);
            
            B=a.*spdiags([.5 -1 .5].*ones(d,1), -1:1, d,d)./(dx^2);
            A=sigma.*spdiags([-.5 0 .5].*ones(d,1), -1:1, d,d)./(dx);
%             A=full(A);
%             B=full(B);
            Xexact=exactHeatEquation1(x,a,sigma);
            X0=[];
            region=floor(d/2-d/(2^(kappa+1))):floor(d/2+d/(2^(kappa+1)));
        otherwise
            error('Unknown example')

    end
    function X=exactHeatEquation1(x,a,sigma)
%         X=zeros(d,d,length(t),M);
        xi=reshape((x(1:1:end-1)+x(2:1:end))./2,[1 d+1 1 1]);
        xsW=x(2:end-1)+sigma.*W1(1,1,:,:);
        Ip=scaled_erf(xi);
        X=...
            (Ip(:,1:1:end-1,:,:)-Ip(:,2:1:end,:,:))./2;
        function Ip=scaled_erf(xi)
            Ip=erf((-xi+xsW)./...
                sqrt(2.*(a-sigma^2).*t));
        end
        X(:,:,1,:)=repmat(eye(d),[1,1,1,M]);
    end
end

