%%*************************************************************************
%% GMMReg:
%% Copyright (c) 2022 by
%% Jingfei Zhang and Yi Li
%%*************************************************************************
%% Input - Z0 centered responses of dimension n*p
%%       - iU n*q design matrix 
%% Note: alpha is fixed at 0.75 for faster computation
%%*************************************************************************

function[beta] = GMMReg(Z0, U)

[n,p]=size(Z0);
iU = [ones(n,1), U]; 
q=size(iU,2)-1;
dim=(p-1)*(q+1);
A=sparse(n*p,p*dim);
b=[];
interM = [];
for j=1:(q+1)
    interM=[interM Z0.*iU(:,j)];
end

%% group structure
int=ceil((1:dim)/(p-1));
G = [];
for i = 1:(q+1)
    G = [G;find(int==i)'];
end
ind = zeros(3,q+1);
grpsize = p-1; 
for i = 1:(q+1)
    if i == 1
        ind(1,1) = 1; ind(2,1) = grpsize; ind(3,1) = 0;
    else
        ind(1,i) = ind(2,i-1) + 1;
        ind(2,i) = i*grpsize;
        ind(3,i) = 1;
    end
end

%% nodewise regression
beta=zeros(p,(p-1)*(q+1));
parfor j=1:p
    A=interM;
    A(:,((0:q)*p+j)) = [];
	y1=Z0(:,j);
	b=y1-mean(y1);
    lam1_max = norm(A'*b,Inf); 
    nl1=50; nl2=1; %nl2=6;
    lambda1  = lam1_max*exp(linspace(log(1), log(0.2), nl1));
    alpha = 0.75; %linspace(0.1, 1, nl2);
    Amap = @(x) mexMatvec(A,x,0);
    ATmap = @(y) mexMatvec(A,y,1);
    AATmap = @(x) Amap(ATmap(x));
    eigsopt = struct();
    eigsopt.issym = 1; 
    eigsopt.tol=1e-3;
    Lip = eigs(AATmap,length(b),1,'LA',eigsopt);
    
    %% cross validation
    Cerror = zeros(nl1,nl2);
    cut = repelem(1:5,n/5); 
    for t=1:5
        foldidx1 = cut~=t;
        Atrain = A(foldidx1,:);
        btrain = b(foldidx1);
        foldidx2 = cut==t;
        Atest = A(foldidx2,:);
        btest = b(foldidx2);
        Amap = @(x) mexMatvec(Atrain,x,0);
        ATmap = @(y) mexMatvec(Atrain,y,1);
        AATmap = @(x) Amap(ATmap(x));
        opts = struct();
        opts.stoptol = 1e-4;
        opts.printyes = 0;
        opts.Lip = Lip;
        Ainput = struct();
        Ainput.A = Atrain;
        Ainput.Amap = @(x) Amap(x);
        Ainput.ATmap = @(x) ATmap(x);
        
        for i=1:nl1
            for k=1:nl2
                lambda2 = lambda1(i)*(1-alpha(k))/alpha(k);
                c = [lambda1(i);lambda2];
                [obj,y,z,x,info,runhist] = SGLasso_SSNAL(Ainput,btrain,size(Atrain,2),c,G,ind,opts);
                res = Atest*x - btest;
                Cerror(i,k) = Cerror(i,k)+sum(res.^2);  
            end
        end
    end

    [id1,id2]=find(Cerror==min(min(Cerror)));
    id1 = id1(1); id2 = id2(1);

    %% estimation
    lambda2 = lambda1(id1)*(1-alpha(id2))/alpha(id2);
    c = [lambda1(id1);lambda2];
    Amap = @(x) mexMatvec(A,x,0);
    ATmap = @(y) mexMatvec(A,y,1);
    AATmap = @(x) Amap(ATmap(x));
    opts = struct();
    opts.Lip = Lip;
    opts.stoptol = 1e-6;
    opts.printyes = 0;
    Ainput = struct();
    Ainput.A = A;
    Ainput.Amap = @(x) Amap(x);
    Ainput.ATmap = @(x) ATmap(x);
    [obj,y,z,x,info,runhist] = SGLasso_SSNAL(Ainput,b,size(A,2),c,G,ind,opts);
    beta(j,:)=x';
end
end
