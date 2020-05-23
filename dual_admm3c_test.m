function [S, X, p_value, time, v] = dual_admm3c_test(X0, G, k, eps, p_iter)
% Wrapper to use dual admm convergent algorithm.
% Args:
%     X0: clustering matrix
%     G:  centered gram matrix
%     k:  number of clusters
%     eps: accuracy tolerence.
%     p_iter:  number of iterations to print out the result
% Returns:
%     S:  dual solution variable
%     X:  primal solution variable
%     p_value: primal value
%     time: time it took to solve the problem
%     v: lagrangian multiplier for the sublevel set constraint.

n = size(X0, 1);
% G = (G + G')/2; % make sure it is symmetric
costmax = trace(G * X0);
max_iter = 10000;

eta = zeros(n); % need to nonnegative
v = 0.01; % need to be nonnegative
S = zeros(n); % need to be psd
X1 = X0;
X2 = X0;
sigma = 0.01;
tau = 1.618;
tic;

[S, X, p_value, v] = dual_admm3c(X0, G, k, costmax, eta, v, S, X1, X2, sigma, tau, max_iter, eps, p_iter);
time=toc;
end
