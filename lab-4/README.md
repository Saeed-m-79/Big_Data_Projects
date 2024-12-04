This lab deals with exploration of networks via sampling and random walks.
Each node n has an attribute, xn, as in the course material. The attributes have been
generated synthetically, in such a way that they are correlated with the node degrees.
Task 1: Exact average
Compute ⟨x⟩.


Task 2: Uniform node sampling
Sample S nodes uniformly at random and estimate ⟨x⟩. Compare your estimate, ⟨x⟩, to the
true value ⟨x⟩.
Run the sampling algorithm five times (with different random seeds), and compare the
output. Comment on the variability in the results.


Task 3: Random neighbor of random node sampling
Sample S nodes by picking a node uniformly at random and examining a random neighbor
of it. Estimate ⟨x⟩ by computing the arithmetic average of the x-values seen this way.
Compare to (i) the true value, and (ii) the theoretical expected value. If the values differ,
explain why and if they are the same, explain why.
Run the sampling algorithm five times (with different random seeds) and compare the
output. Comment on the variability in the results.


Task 4: Uniform random-walk sampling
Sample S nodes by a uniform random walk. Pick a starting node arbitrarily. Run the random
walk for S steps to ensure that the sampler is in “steady-state”. Then run the random walk
for another S steps to collect S x-values.
Estimate ⟨x⟩ by taking the arithmetic average of the x-values seen during the random walk.
Compare to (i) the true value, (ii) the theoretical expected value. If the values differ, explain
why and if they are the same, explain why.
Run the uniform random walk sampler five times (with different random seeds), and compare
the outputs. Comment on the variability in the results.



Task 5: Metropolis-Hastings random walk sampling
Same as task 4, but implement the Metropolis-Hastings sampler instead.
Estimate ⟨x⟩ by taking the arithmetic average of the x-values seen. Compare to (i) the true
value, (ii) the theoretical expected value. If the values differ, explain why and if they are
the same, explain why.
Run the Metropolis-Hastings sampler five times (with different random seeds), and compare
the output. Comment on the variability in the results.
