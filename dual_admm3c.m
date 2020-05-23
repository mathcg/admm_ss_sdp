function [S, X1, p_value, v] = dual_admm3c(X0, G, k, costmax, eta, v, S, X1, X2, sigma, tau, max_iter, eps, p_iter)
% Solve the sublevel set SDP problem:
% min  <X0, X>; such that;  trace(X) = K;  X1 = 1; X >=0; X psd;
%                           <G, X> >= <G, X0>;
% by solving the dual, which is the following SDP problem 
% min kz + alpha^T 1 - v costmax
% subject to; S = C + z I_n + (1 *alpha^T + alpha * 1^T)/2 - beta - v G;
%             S is psd
%             beta >= 0 beta is a n x n matrix
%             v >= 0
% by the algorithm in paper:
%     A Convergent 3-Block Semi-Proximal Alternating Direction Method of 
%     Multipliers for Conic Programming with 4-Type of Constraints.
% 
% Args:
%     X0: clustering matrix
%     G:  centered gram matrix
%     costmax:  <X0, G>
%     eta:  slack variable for the lagrangian multiplier beta for X >=0 constriant;
%     v:  lagrangian multiplier of <G, X> >= <G, X0>, has to be positive;
%     S:  lagrangian multiplier for X is psd;
%     X1: lagrangian multiplier for the dual problem; 
%           S = C + z I_n + (1 *alpha^T + alpha * 1^T)/2 - beta - v G;
%     X2: lagrangian multiplier for the dual problem; eta = beta;
%         beta is the multiplier for the X >= 0 constraint.
%     sigma: penalty term for augmented lagrangian;
%     tau: optimal value for updating agl parameter.
%     max_iter: maximum number of iterations;
%     eps: allowed accuracy term;
%     p_iter:  number of iterations to print out the result
% Returns:
%     S: dual solution variable
%     X1: primal solution variable
%     p_value:  primal value
%     v: lagrangian multiplier for the sublevel set constraint

n = size(X0, 1);

p_value = 0;

% Initialization
% Function argument provides initialization for eta, v, S;
% Then use the update formula for beta, z, alpha to 
% initialize these parameters;

% s_1 = X0 - S - v * G + eta;
% a_1 = -trace(X0 - S - v * G);
% b_1 = -(X0 - S - v * G) * ones(n, 1);
% 
% sum_alpha = (sum(sum(s_1)) + 2 * sum(b_1) - trace(s_1) - 2 * a_1)/(n-1);
% z = (trace(s_1) + 2 * a_1 - sum_alpha)/n;
% alpha = (2 * s_1 * ones(n, 1) + 4 * b_1 - sum_alpha * ones(n, 1) - 2 * z * ones(n, 1))/n;
% A_alpha = (ones(n, 1) * alpha' + alpha * ones(1, n))/2;
% 
% beta = 0.5 * (s_1 + z * eye(n) + A_alpha);

sum_alpha = (sum(sum(eta + S)) - trace(eta + S + v * G) + k - n)/(n-1);
z = (trace(S + v * G + eta) - k - sum_alpha)/n;
alpha = (2 * (S + eta) * ones(n, 1) - 2 - sum_alpha - 2 * z)/n;
A_alpha = (ones(n, 1) * alpha' + alpha * ones(1, n))/2;
beta = 0.5 * (X0 - S - v * G + eta + A_alpha + z * eye(n));

% record number of positive eigenvalues as we proceed;
num_pos = zeros(1, max_iter);

norm_G = norm(G, 'fro')^2;

for i = 1:max_iter
    
% Update S and eta variable
T_1 = v * G - z*eye(n) - A_alpha + beta - X0;
M = -T_1 - X1/sigma;

[v_ignore, S] = proj_psd_largescale(1, -M, 1);
S = S + M;
S = (S + S')/2;

eta = max(beta - X2/sigma, 0);

% Update beta, z and alpha variable
sum_alpha = (sum(sum(eta + S)) - trace(eta + S + v * G) + k - n)/(n-1);
z = (trace(S + v * G + eta) - k - sum_alpha)/n;
alpha = (2 * (S + eta) * ones(n, 1) - 2 - sum_alpha - 2 * z)/n;
A_alpha = (ones(n, 1) * alpha' + alpha * ones(1, n))/2;
beta = 0.5 * (X0 - S - v * G + eta + A_alpha + z * eye(n));

% Update v variable
T_2 = S - z * eye(n) - A_alpha + beta - X0;
v = (costmax - trace(G * X1) - sigma * trace(G * T_2)) / (sigma * norm_G);
v = max(v, 0);

% Update beta, z and alpha variable again
sum_alpha = (sum(sum(eta + S)) - trace(eta + S + v * G) + k - n)/(n-1);
z = (trace(S + v * G + eta) - k - sum_alpha)/n;
alpha = (2 * (S + eta) * ones(n, 1) - 2 - sum_alpha - 2 * z)/n;
A_alpha = (ones(n, 1) * alpha' + alpha * ones(1, n))/2;
beta = 0.5 * (X0 - S - v * G + eta + A_alpha + z * eye(n));

% Update X1 and X2 variable
X1 = X1 + tau * sigma * (S + v * G + beta - A_alpha - z * eye(n) - X0);
X2 = X2 + tau * sigma * (eta - beta);

p_value = trace(X0 * X1);
dual_value = k * z + sum(alpha) - v * costmax;

relative_gap = abs(p_value - dual_value) / (1 + abs(p_value) + abs(dual_value));

% compute constraint violation
% compute primal infeasibility
eta_e = 0; % by our algorithm, the equality constraint always satisfy.
eta_i = abs(min(real(trace(G * X1) - costmax), 0)) / (1 + abs(costmax));
eta_P = max(eta_e, eta_i);

% compute dual infeasibility
eta_D = norm(S + v * G + beta - A_alpha - z * eye(n) - X0, 'fro') / (1 + sqrt(k));

eta = max(eta_D, eta_P);
if mod(i, p_iter) == 0 || eta <= eps
    fprintf('after %d iteration, p_value is %f\n', i, p_value)
    fprintf('after %d iteration, dual value is %f\n', i, -dual_value)
%     fprintf('primal equality violation is %f, inequality violation is %f\n', eta_e, eta_i);
    fprintf('the dual violation is %f; the primal violation is %f\n', eta_D, eta_P);
    fprintf('primal nonnegativity violation is %f\n', abs(min(X1(:))));
    fprintf('\n');
end


if eta <= eps
    fprintf('the multiplier value sigma is %f\n', sigma);
    break
end

end
