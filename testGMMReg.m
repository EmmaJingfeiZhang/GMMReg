clear; rng('default');
addpath(genpath(pwd));


n= 100; % sample size
p = 50; % dimension of response
q = 50; % dimension of covariates
dim=(p-1)*(q+1); % dimension in each nodewise regression


%% true coefficient represented as a p*p*(q+1) array
tB = tensor(zeros(1,p*p*(q+1)), [p p q+1]); 
for i=2:4
	for j=[1,3,5]
		tB(j,j+1,i) = 0.3;
        tB(j+1,j,i) = tB(j,j+1,i);
    end
end

[Z U] = X_simulate(n,p,q,tB); %simulate data
beta = GMMReg(Z, U); %Gaussian graphical estimation


%% organize the coefficient into a p*p*(q+1) array
BB = zeros(p,p,q+1);
for j=1:p
    BB(setdiff(1:p,j),j,:) = reshape(beta(j,:)',p-1,q+1);
end
Bhat = zeros(p,p,q+1);
for j=1:(q+1)
    Bhat(:,:,j)= BB(:,:,j).*(abs(BB(:,:,j))<abs(BB(:,:,j)'))+BB(:,:,j)'.*(abs(BB(:,:,j)')<abs(BB(:,:,j)));
end

%% find TPR and FPR
tB = double(tB);
tpr = sum((Bhat~=0).*(tB~=0),'all')/sum(tB~=0,'all'); 
fpr = sum((Bhat~=0).*(tB==0),'all')/(sum((Bhat~=0).*(tB==0),'all')+sum(tB==0,'all'));



