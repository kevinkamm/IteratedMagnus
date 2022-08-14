function [A,B,X0,Xexact,region1,region2,EulerLangevin]=coefficients(t,M,d,dW1,example)
    N=length(t);
    t=reshape(t,1,1,[],1);
    T=t(end);
    dt=T/(N-1);
    W1=zeros(size(dW1)+[0 0 1 0]);
    W1(1,1,2:end,:)=dW1;
    W1=cumsum(W1,3);
%     bdFunc=@(x) 1./(1+exp(-x));%sigmoid
%     bdFunc=@(x) 5.*exp(-x.^2);%Gauss
%     bdFunc=@(x) 2.*(1/(1+x.^2));
%     bdFunc=@(x) 2.*atan(x);
%     bdFunct=@(t,x) 2.*atan(x).*t;
    kappa=2;
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
            region1=floor(d/2-d/(2^(kappa+1))):floor(d/2+d/(2^(kappa+1)));
            region2=region1;
        case 'Tridiag2x2exm1'
%             A=rand(2,2);
%             B=rand(2,2);
%             A(2,1)=0;
%             B(2,1)=0;
%             disp(A);
%             disp(B);
            A=[0.089169311678098,0.401498312870165;0,0.016873613318980];
            B=[0.923389293105627,0.381331107923466;0,0.540312021308851];
            disp(A);
            disp(B);
%             A=A./norm(A,'fro');
%             B=B./norm(A,'fro');
            X0=[];
            Xexact=exactTriDiag2x2(X0,A,B);
            region1=1;
            region2=2;
        case 'Langevin1'
            x=linspace(-4,4,d+2)';
            v=linspace(-4,4,d+2);
            a=1.1;
            sigma=1/sqrt(10);
            s = 1;
            phi = @(x,v) exp(-(x.^2+v.^2)./(2.*s));%./(2.*pi.*s);
            h = @(x,v) 0;
            fx = @(x,v) -v;
            fv = @(x,v) 0;
            gxx = @(x,v) 0;
            gvv = @(x,v) a;
            gxv = @(x,v) 0;
            sx = @(x,v) 0;
            sv = @(x,v) sigma;
            [A,B,Dx,Dv,Dxx,Dvv,H,Fx,Fv,Gxx,Gxv,Gvv,Sx,Sv,Phi]=Langevin(x,v,h,fx,fv,gxx,gxv,gvv,sx,sv,phi);
            EulerLangevin={};
            EulerLangevin{end+1}=Dx;
            EulerLangevin{end+1}=Dv;
            EulerLangevin{end+1}=Dxx;
            EulerLangevin{end+1}=Dvv;
            EulerLangevin{end+1}=H;
            EulerLangevin{end+1}=Fx;
            EulerLangevin{end+1}=Fv;
            EulerLangevin{end+1}=Gxx;
            EulerLangevin{end+1}=Gxv;
            EulerLangevin{end+1}=Gvv;
            EulerLangevin{end+1}=Sx;
            EulerLangevin{end+1}=Sv;
            EulerLangevin{end+1}=repmat(Phi,1,1,1,M);
            X0=Phi(:);
            region1=floor(d/2-d/(2^(kappa+1))):floor(d/2+d/(2^(kappa+1)));
            region2=region1;
            Xexact=LangevinConstExact(T,x(2:end-1),v(2:end-1),a,sigma,W1);
        otherwise
            error('Unknown example')

    end
    function X=exactTriDiag2x2(X0,A,B)
        if isempty(X0)
            X0=[1 0; 0 1];
        end
        X=zeros(2,2,1,M);
        Ia2ds=A.^2.*t;
        Ibds=B.*t;
        IadWs=A.*W1;
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

        EZinv=exp(-(Ibds(1,1,:,:)+IadWs(1,1,:,:)-Ia2ds(1,1,:,:)./2));
        dHLeb=B(1,2).*X22temp;
        dHStoch=A(1,2).*X22temp;
        dHZ=-dHStoch.*A(1,1);
        X(1,2,:,:)=EZ.*(X0(1,2)+...
                    lebesgueInt(EZinv.*(dHLeb+dHZ),dt,T)+...
                    stochInt(EZinv.*dHStoch,dW1));
        
    end
    function X=exactHeatEquation1(x,a,sigma)
        X=zeros(d,d,1,M);
        xi=reshape((x(1:1:end-1)+x(2:1:end))./2,[1 d+1 1 1]);
        xsW=x(2:end-1)+sigma.*W1(1,1,end,:);
        Ip=scaled_erf(xi);
        X(:,:,1,:)=...
            (Ip(:,1:1:end-1,:,:)-Ip(:,2:1:end,:,:))./2;
        function Ip=scaled_erf(xi)
            Ip=erf((-xi+xsW)./...
                sqrt(2.*(a-sigma^2).*T));
        end
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
function [A,B,Dx,Dv,Dxx,Dvv,H,Fx,Fv,Gxx,Gxv,Gvv,Sx,Sv,Phi]=Langevin(xgrid,vgrid,h,fx,fv,gxx,gxv,gvv,sx,sv,phi)
%%COEFFICIENTS computes the coefficients for Euler and Magnus.
%   Input:
%       xgrid (nx+2 x 1 array): position grid
%       vgrid (1 x nv+2 array): velocity grid
%       h (function handle, args x,v): drift coefficient for zero
%                                      derivative
%       fx (function handle, args x,v): drift coefficient for first 
%                                       derivative wrt x
%       fv (function handle, args x,v): drift coefficient for first 
%                                       derivative wrt v
%       gxx (function handle, args x,v): drift coefficient for second 
%                                        derivative wrt x
%       gvv (function handle, args x,v): drift coefficient for second 
%                                        derivative wrt x
%       gxv (function handle, args x,v): drift coefficient for second 
%                                        derivative wrt x and v
%       sx (function handle, args x,v): diffusion coefficient for first 
%                                       derivative wrt x
%       sv (function handle, args x,v): diffusion coefficient for first 
%                                       derivative wrt v
%   Output:
%       A (nx*nv x nx*nv sparse array): Magnus coefficient for Ito integral
%       B (nx*nv x nx*nv sparse array): Magnus coefficient for Lebesgue integral
%       Dx (nx x nx sparse array): first derivative wrt x (central diff)
%       Dv (nv x nv sparse array): first derivative wrt v (central diff)
%       Dxx (nv x nv sparse array): second derivative wrt x (central diff)
%       Dvv (nv x nv sparse array): second derivative wrt v (central diff)
%       H
%       Fx
%       Fv
%       Gxx
%       Gxv
%       Gvv
%       Sx
%       Sv
%
%   Usage:
%       [A,B,Dx,Dv,Dxx,Dvv,H,Fx,Fv,Gxx,Gxv,Gvv,Sx,Sv]=
%           coefficients(xgrid,vgrid,h,fx,fv,gxx,gxv,gvv,sx,sv)
%

nx=length(xgrid)-2; %subtract boundary conditions
nv=length(vgrid)-2; %subtract boundary conditions
dx=(xgrid(end)-xgrid(1))/(length(xgrid)-1);
dv=(vgrid(end)-vgrid(1))/(length(vgrid)-1);

Dx=0.5.*(1./dx).*spdiags([-1,0,1].*ones(nx,1),-1:1,nx,nx);
Dv=0.5.*(1./dv).*spdiags([-1,0,1].*ones(nv,1),-1:1,nv,nv);
Dxx=(1./(dx.^2)).*spdiags([1,-2,1].*ones(nx,1),-1:1,nx,nx);
Dvv=(1./(dv.^2)).*spdiags([1,-2,1].*ones(nv,1),-1:1,nv,nv);
H=broadcast(nx,nv,h(xgrid(2:end-1),vgrid(2:end-1)));
Fx=broadcast(nx,nv,fx(xgrid(2:end-1),vgrid(2:end-1)));
Fv=broadcast(nx,nv,fv(xgrid(2:end-1),vgrid(2:end-1)));
Gxx=broadcast(nx,nv,gxx(xgrid(2:end-1),vgrid(2:end-1)));
Gxv=broadcast(nx,nv,gxv(xgrid(2:end-1),vgrid(2:end-1)));
Gvv=broadcast(nx,nv,gvv(xgrid(2:end-1),vgrid(2:end-1)));
Sx=broadcast(nx,nv,sx(xgrid(2:end-1),vgrid(2:end-1)));
Sv=broadcast(nx,nv,sv(xgrid(2:end-1),vgrid(2:end-1)));
Phi=broadcast(nx,nv,phi(xgrid(2:end-1),vgrid(2:end-1)));
Ix=speye(nx,nx);
Iv=speye(nv,nv);
B=largeFullSparseMult(Fx(:),kron(Iv,Dx))+...
    largeFullSparseMult(Fv(:),kron(Dv,Ix))+...
    largeFullSparseMult(Gxx(:)./2,kron(Iv,Dxx))+...
    largeFullSparseMult(Gxv(:),kron(Dv,Dx))+...
    largeFullSparseMult(Gvv(:)./2,kron(Dvv,Ix));
if numel(H)==1
    H=H.*ones(nx,nv);
end
B=B+spdiags(H(:),1,nx*nv,nx*nv);
    
A=largeFullSparseMult(Sx(:),kron(Iv,Dx))+...
    largeFullSparseMult(Sv(:),kron(Dv,Ix));
end
function Y=broadcast(nx,nv,X)
%     if size(X,1)==1 && size(X,2)==1
%         Y=X.*ones(nx,nv);
    if size(X,1)==1 && size(X,2)==nv
        Y=ones(nx,1).*X;
    elseif size(X,1)==nx && size(X,2)==1
        Y=X.*ones(1,nv);
    else
        Y=X;
    end
end
function Y=largeFullSparseMult(v,X)
    if numel(v)==1
        Y=v.*X;
    else
%         Y=gather(distributed(v).*X);
        Y=spdiags(v,0,size(X,1),size(X,2))*X;
    end
end
function utExact=LangevinConstExact(T,xgrid,vgrid,a,sigma,W)
    N=size(W,3);
    M=size(W,4);
    
    dt=T/(N-1);
    IW=l_int(W,dt);
    
    x=xgrid;
    v=vgrid;
    t=T;
    Wt=W(1,1,end,:);
    It=IW(1,1,1,:);
    utExact=...
        2.*3.^(1/2).*exp((-2).*(12+(-12).*sigma.^2.*t+(-4).* ...
          sigma.^2.*t.^3+a.^2.*t.^4+sigma.^4.*t.^4+a.*(12.*t+4.*t.^3+(-2).* ...
          sigma.^2.*t.^4)).^(-1).*((3+3.*t.^2+(a+(-1).*sigma.^2).*t.^3).* ...
          v.^2+sigma.^2.*(3.*It.^2.*(1+a.*t+(-1).*sigma.^2.*t)+(-3).*It.*t.* ...
          (2+a.*t+(-1).*sigma.^2.*t).*Wt+(3+3.*t.^2+(a+(-1).*sigma.^2).* ...
          t.^3).*Wt.^2)+3.*sigma.*(2.*It.*(1+a.*t+(-1).*sigma.^2.*t)+t.*(( ...
          -2)+(-1).*a.*t+sigma.^2.*t).*Wt).*x+3.*(1+a.*t+(-1).*sigma.^2.*t) ...
          .*x.^2+v.*(6.*sigma.*Wt+2.*sigma.*(a+(-1).*sigma.^2).*t.^3.*Wt+( ...
          -6).*t.*(It.*sigma+x)+t.^2.*((-3).*a.*(It.*sigma+x)+3.*sigma.*(2.* ...
          Wt+sigma.*(It.*sigma+x)))))).*(12+(-12).*sigma.^2.*t+(-4).* ...
          sigma.^2.*t.^3+a.^2.*t.^4+sigma.^4.*t.^4+a.*(12.*t+4.*t.^3+(-2).* ...
          sigma.^2.*t.^4)).^(-1/2);
end
function I=l_int(W,dt)
    I=sum(W(1,1,1:1:end-1,:).*dt,3); 
end
