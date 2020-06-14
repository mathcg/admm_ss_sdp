function [P, epsilon, obj_value, time] = cg_ss_test(X0, G, K, max_iter, p_iter)
% Helper function to run conditional gradient for sublevel set SDP
% problem.
% Args:
%     X0:  clustering matrix.
%     G:   gram matrix;  data are first centered, then a gram matrix with
%          element being x_i^T x_j was computed;
%     K:   number of clusters to compare with
%     max_iter:  maximum number of iterations conditional gradient should
%                run
%     p_iter: number of iterations to print out result
% Returns:
%     P:   minimizer of sublevel set SDP problem
%     epsilon:  minimum value of sublevel set SDP problem
%     time:  time it took to solve the problem.

x2 = -0.01; 
N_inner = 10;

v_max = eigs(G, 1, 'la');  % normalize matrix G
v_max = v_max/10;
G_norm = G/v_max;

[u, v] = eig(G_norm);  % get the spectral representation of matrix G
eig_loc = find(diag(v) > 1e-10);
G_u_temp = u(:, eig_loc) * sqrt(v(eig_loc, eig_loc));

[u, v] = eig(X0);  % get the spectral representation of matrix X0
eig_loc = find(diag(v) > 1e-10);
X0_u_temp = u(:, eig_loc) * sqrt(v(eig_loc, eig_loc));
temp = G_u_temp' * X0_u_temp;
costmax = norm(temp, 'fro')^2;

[P, epsilon, obj_value, time] = cg_ss(X0, G_u_temp, X0_u_temp, K, costmax,... 
                           x2, max_iter, N_inner, p_iter);
end