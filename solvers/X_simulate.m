function[X,U] = X_simulate(n,p,q,tB)

N = randi([0 1],n,q);
U = normalize(N);
iU = [ones(n,1), U];
X = zeros(n, p);
for i=1:n
    omega = ttv(tB,iU(i,:)', 3);
    omega = double(omega);
    omega = omega + diag(ones(1,p));
    sigma = inv(omega);
    X(i,:) = mvnrnd(zeros(1,p),sigma,1);
end
end
