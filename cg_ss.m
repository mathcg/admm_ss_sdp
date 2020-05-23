function [P, epsilon, time] = cg_ss(X0, G_half, X0_half, K, costmax, x2, max_iter, N_inner, p_iter)
% use conditional gradient method to solve the sublevel set problem
% The algorithm following as first rewriting the problem as 
% min <X0, P> s.t.  P1 = 0, tr(P) = K - 1;  P psd;  P + E_n >= 0;  <G, P>
% >= <G, X0> - <G, E_n>;
% So here we are using P = X - E_n with E_n = 11'/n; Then we use augmented
% lagrangian method to solve the problem
% The augmented Lagrangian with respect to constraint P + E_n >= 0 and
% <G, P> >= <G, X0> - <G, E_n> is as following:
% g(P, Gamma, X2) = <X0, P> + <Gamma, P + E_n> + gamma/2 * |(P + E_n)_|_F^2
% + X_2 ( <G, P> - <G, X0> + <G, E_n>) + gamma/2 * (<G, P> - <G, X0> + <G,
% E_n)_^2
% Here, parameter costmax = <G, X0> - <G, E_n>

% Args:
%     X0:  clustering matrix.
%     G_half:  spectral of G, since data are lower dimensional (n >> d), 
%              this makes the matrix multiplication involving G faster.
%     X0_half: spectral of X0, again since n >> K;
%     K:   number of clusters to compare with
%     costmax:   <G, X0> - <G, E_n>
%     x2:  initialization value for lagrangian multiplier x2.
%     max_iter:  maximum number of iterations conditional gradient should
%                run
%     N_inner:  number of inner iterations.
%     p_iter: number of iterations to print out result
% Returns:
%     P:   minimizer of sublevel set SDP problem
%     epsilon:  minimum value of sublevel set SDP problem
%     time:  time it took to solve the problem.

n = size(X0, 1);
P = zeros(n);
trace_G_P = 0;
trace_X0_P = 0;
Gamma = zeros(n);
gamma = 1;
tau = 1.618;

one_v = ones(n, 1);
one_over_n = 1./n;

opts.isreal = 1;
opts.issym = 1;

tic
for i = 1:max_iter
    for j = 1: N_inner
        P_plus_En = P + one_over_n;
        G_P_minus = trace_G_P - costmax;
        
        x2_temp = x2 + gamma * min(G_P_minus, 0);
        g_grad = Gamma + gamma * min(P_plus_En, 0);
        temp = 1/n * (g_grad * one_v);
        g_grad_mean = mean(g_grad(:)) + 1/n;  % 1/n since the mean of X0 is 1/n;
        
        B1fun = @(x) sum(x) * (temp - g_grad_mean + 2/n) + temp' * x;
        C1fun = @(x) X0_half * (X0_half' * x) + x2_temp * (G_half * (G_half' * x));
        A1fun = @(x) g_grad * x + C1fun(x) - B1fun(x);

        opts.tol = 1/j;
        [u, v_ignore] = eigs(A1fun, n, 1, 'sa', opts);
        if v_ignore > 0
%             fprintf('positive eigenvalue for A\n');
            continue
        end
        H = (K - 1) *  (u * u');
%         alpha = 2/((i-1)*N_inner + j - 1 + 2);
        alpha = 2/(i + j + 2);
        P = (1 - alpha) * P + alpha * H;
        trace_G_P = (1 - alpha) * trace_G_P + (K - 1) * alpha * norm(G_half' * u)^2;
        trace_X0_P = (1 - alpha) * trace_X0_P + (K - 1) * alpha * norm(X0_half' * u)^2;
    end

    P_nneg = P + one_over_n;
    Gamma = min(Gamma + tau * gamma * P_nneg, 0);
    x2 = min(x2 + gamma * (trace_G_P - costmax), 0);
    max_error = abs(min(P_nneg(:)));
    
    if mod(i, p_iter) == 0 || max_error <= 1e-5
        obj_value = trace_X0_P + 1;
        fprintf('after %d iteration, cost difference is %f\n', i, trace_G_P - costmax);
        fprintf('after %d iteration, max_error is %f\n', i, max_error)
        fprintf('after %d iteration, obj value is %f\n', i, obj_value)
        fprintf('\n');
    end
    if max_error <= 1e-5
        break
    end
end

P = P + one_over_n;
time = toc;
epsilon = trace_X0_P + 1;
fprintf('final objective would be %f\n', epsilon);
