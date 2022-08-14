function [A,a,B1,B2,b,X0,Xexact,region1,region2,EulerLangevin]=coefficients(t,M,d,example)
    N=length(t);
    t=reshape(t,1,1,[],1);
    T=t(end);
    dt=T/(N-1);
%     bdFunc=@(x) 1./(1+exp(-x));%sigmoid
%     bdFunc=@(x) 2.*exp(-x.^2);%Gauss
%     bdFunc=@(x) 2.*atan(x);
    bdFunc=@(x) (1+1/(1+x.^2));
    bdFunctB=@(t,x) bdFunc(x);
    bdFunctA=@(t,x) sqrt(bdFunc(x));
%     bdFunct=@(t,x) bdFunc(x);
    kappa=4;
    EulerLangevin={};
    switch example
        case 'Langevin1'
            dW2=sqrt(dt).*randn(1,1,N-1,M);
            W2=zeros(1,1,N,M);
            W2(1,1,2:end,:)=dW2;
            W2=cumsum(W2,3);
            x=linspace(-4,4,d+2)';
            v=linspace(-4,4,d+2);
            aC=1.1;
            sigmaC=1/sqrt(10);
            s = 1;
            phi = @(x,v) exp(-(x.^2+v.^2)./(2.*s));%./(2.*pi.*s);
%             h = @(x,v) 0;
            fx = @(x,v) -v;
%             fv = @(x,v) 0;
%             gxx = @(x,v) 0;
            gvv = @(x,v) aC;
%             gxv = @(x,v) 0;
%             sx = @(x,v) 0;
            sv = @(x,v) sigmaC;
            [A,B1,B2,Dx,Dv,Dxx,Dvv,Fx,Gvv,Sv,Phi]=Langevin(x,v,fx,gvv,sv,phi);
            Sx=0;
            Gxv=0;
            Gxx=0;
            Fv=0;
            H=0;
            a=bdFunctA(t,W2(1,1,:,:));
            b=bdFunctB(t,W2(1,1,:,:));
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
            Xexact=[];
        otherwise
            error('Unknown example')

    end
end

function [A,B1,B2,Dx,Dv,Dxx,Dvv,Fx,Gvv,Sv,Phi]=Langevin(xgrid,vgrid,fx,gvv,sv,phi)

nx=length(xgrid)-2; %subtract boundary conditions
nv=length(vgrid)-2; %subtract boundary conditions
dx=(xgrid(end)-xgrid(1))/(length(xgrid)-1);
dv=(vgrid(end)-vgrid(1))/(length(vgrid)-1);

Dx=0.5.*(1./dx).*spdiags([-1,0,1].*ones(nx,1),-1:1,nx,nx);
Dv=0.5.*(1./dv).*spdiags([-1,0,1].*ones(nv,1),-1:1,nv,nv);
Dxx=(1./(dx.^2)).*spdiags([1,-2,1].*ones(nx,1),-1:1,nx,nx);
Dvv=(1./(dv.^2)).*spdiags([1,-2,1].*ones(nv,1),-1:1,nv,nv);

Fx=broadcast(nx,nv,fx(xgrid(2:end-1),vgrid(2:end-1)));
Gvv=broadcast(nx,nv,gvv(xgrid(2:end-1),vgrid(2:end-1)));
Sv=broadcast(nx,nv,sv(xgrid(2:end-1),vgrid(2:end-1)));
Phi=broadcast(nx,nv,phi(xgrid(2:end-1),vgrid(2:end-1)));

Ix=speye(nx,nx);
Iv=speye(nv,nv);

B1=largeFullSparseMult(Fx(:),kron(Iv,Dx));
B2=largeFullSparseMult(Gvv(:)./2,kron(Dvv,Ix));
    
A=largeFullSparseMult(Sv(:),kron(Dv,Ix));
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