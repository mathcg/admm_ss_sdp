# admm_ss_sdp
This repo contains two solvers for solving the sublevel set SDP problem (Marina 2018) for k-means clustering. 

`dual_admm3c.m` impletements the ADMM algorithm for semidefinite programming problem based on the paper (Sun 2015). It is a specialized version for the sublevel set SDP problem. It is suitable for smaller problems with n <= 1000. `dual_admm3c_test.m` is a wrapper function to use the solver.  `proj_psd_largescale` is taken from library [TFOCS](https://github.com/cvxr/TFOCS/blob/master/proj_psd.m) to project matrix onto semidefinite cone. 

`cg_ss.m` extends the conditional gradient algorithm (Tepper 2018) for k-means SDP problem to the sublevel set SDP problem. It is suitable for problems with large n. For smaller problems, conditional gradient sometimes performs worse than the ADMM algorithm. `cg_ss_test.m` is a wrapper function to use the solver.

# Sample Usage
```matlab
load X0_200
[S, X, p_value, time, v] = dual_admm3c_test(X0, G, 4, 1e-4, 100);
[P, epsilon, obj_value, time] = cg_ss_test(X0, G, 4, 5000, 50);
```

# References

Sun, D., Toh, K. C., & Yang, L. (2015). A convergent 3-block semiproximal alternating direction method of multipliers for conic programming with 4-type constraints. SIAM journal on Optimization, 25(2), 882-915.

Meila, M. (2018). How to tell when a clustering is (approximately) correct using convex relaxations. In Advances in Neural Information Processing Systems (pp. 7407-7418).

Tepper, M., Sengupta, A. M., & Chklovskii, D. (2018). Clustering is semidefinitely not that hard: Nonnegative SDP for manifold disentangling. The Journal of Machine Learning Research, 19(1), 3208-3237.
