 Objective:
 
 The task in this lab exercise is to implement spectral modularity maximization for bisection
 of a network into two communities of a priori unknown size.
 Wewill work on four networks:
 1. A small test network for which we provide a reference solution (for debugging
 purposes)
 2. The karate club network
 3. Asmall“stochastic block model” (synthetic data with community structure) with two
 communities
 4. Alarge “stochastic block model” (synthetic data with community structure) with two
 communities, for which computational aspects are crucial
 All networks can be loaded via the function load_data.py. (Uncomment the pertinent
 lines in the code to get the desired network.

Task 1

 Implement spectral modularity maximization using the power method. For networks 1–3,
 “normal” linear algebra routines are sufficient. For network 4, use the Python package
 scipy.sparse for numerical linear algebra with sparse matrices.
 To debug your solution you can compare to the reference solution output below for the test
 network.
 
 Task 2
 
 Run the community detection algorithm on the Zachary karate club network. Layout the
 network in Gephi with ForceAtlas and comment on the result. Make sure to import the
 community labels as well. Use for example the functions in save_Gephi_gexf.py and/or
 save_Gephi_gexf.m to do that. These functions make sure that Gephi colors the nodes
 according to the community labels identified by your algorithm. Does the community
 partitioning look intuitive?
 
 Task 3
 
 Run the community detection algorithm on the small synthetic stochastic block model
 network. Layout the network in Gephi with ForceAtlas and comment on the result. Does
 the community partitioning look intuitive? Again, if you use save_Gephi_gexf.py or
 save_Gephi_gexf.m to save your result then Gephi will color the nodes according to the
 community partitioning that you have identified.
 Also compare the result with Gephi’s community partitioning, and compare the modularity
 score of your solution with that given by Gephi. (Gephi uses a different algorithm!)
 
 Task 4
 
 Run your community detection algorithm on the large synthetic stochastic block model
 network. How long did it take to run?
 For comparison, construct the modularity matrix (𝒁 in the lecture) explicitly, and try to use
 numpy to find its eigenvalues. (The computer may “hang”. Why?)
