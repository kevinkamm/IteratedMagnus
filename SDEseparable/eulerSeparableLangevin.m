function ut=eulerSeparableLangevin(T,dWeuler,a,b,Coeffs,varargin)
%%EULER computes the Euler-Maruyama scheme for the stochastic Langevin
% equation with constant coefficients.
% Magnus (A, B).
%   Input:
%       D (nv x nv sparse array): -v part in SPDE
%       E (nx x nx sparse array): first derivative wrt x (central diff)
%       F (nv x nv sparse array): first derivative wrt v (central diff)
%       G (nv x nv sparse array): second derivative wrt v (central diff)
%       varargin (cell array): name value pairs
%           'Device' (default: cpu): 'cpu', 'gpu'
%           'Mode' (default: full): 'sparse', 'full'
%   Output:

[Dx,Dv,Dxx,Dvv,H,Fx,Fv,Gxx,Gxv,Gvv,Sx,Sv,phi]=Coeffs{:};

device='cpu';
mode='full';
if isempty(varargin)==0
    for k=1:2:length(varargin)
        switch varargin{k}
            case 'device'
                device=varargin{k+1};
            case 'mode'
                mode=varargin{k+1};
        end
    end
end
M=size(dWeuler,4);
Neuler=size(dWeuler,3)+1;
nx=size(Dx,1);
nv=size(Dv,1);
dt=T/(Neuler-1);
switch mode
    case 'full'
        Dx=full(Dx);
        Dv=full(Dv);
        Dxx=full(Dxx);
        Dvv=full(Dvv);
        H=full(H);
        Fx=full(Fx);
        Fv=full(Fv);
        Gxx=full(Gxx);
        Gxv=full(Gxv);
        Gvv=full(Gvv);
        Sx=full(Sx);
        Sv=full(Sv);
    case 'sparse'
    otherwise
        error("Unsupported mode: " +...
               mode + " given; full or sparse expected");
end

switch device
    case 'gpu'
        Dx=gpuArray(Dx);
        Dv=gpuArray(Dv);
        Dxx=gpuArray(Dxx);
        Dvv=gpuArray(Dvv);
        H=gpuArray(H);
        Fx=gpuArray(Fx);
        Fv=gpuArray(Fv);
        Gxx=gpuArray(Gxx);
        Gxv=gpuArray(Gxv);
        Gvv=gpuArray(Gvv);
        Sx=gpuArray(Sx);
        Sv=gpuArray(Sv);
%         dWeuler=gpuArray(dWeuler);
%         ut=gpuArray.zeros(nx,nv,Neuler,M);
%         ut(:,:,1,:)=repmat(gpuArray(phi),[1,1,1,M]);
        ut=zeros(nx,nv,1,M);
        ut(:,:,1,:)=phi;
    case 'cpu'
        ut=zeros(nx,nv,1,M);
        ut(:,:,1,:)=phi;
    otherwise
        error("Unsupported device: " +...
               device + " given; cpu or gpu expected");
end

for tk=1:1:Neuler-1
    ut=ut+...
        (H.*ut+...
        Fx.*leftMult(Dx,ut)+...
        Fv.*rightMult(ut,Dv')+...
        (Gxx./2).*leftMult(Dxx,ut)+...
        Gxv.*rightMult(leftMult(Dx,ut),Dv')+...
        (Gvv./2).*rightMult(ut,Dvv')).*dt.*b(1,1,tk,:)+...
        (Sx.*leftMult(Dx,ut)+...
        Sv.*rightMult(ut,Dv')).*dWeuler(:,:,tk,:).*a(1,1,tk,:);
end

function Z=leftMult(A,X)
    switch string(device)+" "+string(mode)
        case "gpu full"
            X=gpuArray(X);
%             Z=pagefun(@mtimes,A,X); 
            Z=pagemtimes(A,X);
        case "cpu full"
            Z=pagemtimes(A,X);
        case "cpu sparse"
            Z=zeros(nx,nv,1,M);
            for mi=1:1:M
                Z(:,:,1,mi)=A*X(:,:,mi);
            end
        case "gpu sparse"
            X=gpuArray(X);
            Z=gpuArray.zeros(nx,nv,1,M);
            for mi=1:1:M
                Z(:,:,1,mi)=A*X(:,:,mi);
            end
        otherwise
            error('Unknown device or mode')
    end
end
function Z=rightMult(X,A)
    switch string(device)+" "+string(mode)
        case "gpu full"
            X=gpuArray(X);
%             Z=pagefun(@mtimes,X,A); 
            Z=pagemtimes(X,A);
        case "cpu full"
            Z=pagemtimes(X,A);
        case "cpu sparse"
            Z=zeros(nx,nv,1,M);
            for mi=1:1:M
                Z(:,:,1,mi)=X(:,:,mi)*A;
            end
        case "gpu sparse"
            X=gpuArray(X);
            Z=gpuArray.zeros(nx,nv,1,M);
            for mi=1:1:M
                Z(:,:,1,mi)=X(:,:,mi)*A;
            end
        otherwise
            error('Unknown device or mode')
    end
end
end