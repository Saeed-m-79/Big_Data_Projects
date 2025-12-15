
import sys
import snap
import numpy as np
sys.path.append("/courses/TSKS33/ht2024/common-functions")
from save_Gephi_gexf import saveGephi_gexf
from save_Gephi_gexf import saveGephi_gexf_twocolors
from gen_stochblock import gen_stoch_block_2comm
import snap_scipy as sp
from save_Gephi_gexf import save_csrmatrix_Gephi_gexf_twocolors
from scipy.sparse import diags
G = snap.LoadEdgeList(snap.PUNGraph, "test-network.txt", 0, 1)

#G = snap.LoadEdgeList(snap.PUNGraph, "karate-network.txt", 0, 1)


#G = snap.LoadEdgeList(snap.PUNGraph, "SB-small-network.txt", 0, 1)
#G = snap.LoadEdgeList(snap.PUNGraph, "SB-large-network.txt", 0, 1)

#snap.SaveMatlabSparseMtx(G, "test-network.mat")

A_sparse =sp.to_sparse_mat(G)
#print(A)

K = A_sparse.sum(axis=1).A.flatten()  # .A converts the result to a dense array

# Step 2: Create a sparse diagonal matrix from the degree vector
K_sparse = diags(K)  # Creates a sparse diagonal matrix with the degrees

# Now K_sparse is a sparse degree matrix, and you can use it efficiently
print(K_sparse)


print(f"{K} is the degree matrix")
m = K_sparse.sum(axis = 0)
print(f"twice the number of edges {m}")
print(f"the sparse representation of the degree matrix : {K_sparse} ")


#from s5 import solve_test_network

#x = solve_test_network()

total_edges = K_sparse.sum()
#csr_matrix
print(f"total edges : {total_edges}")

#K_outer_sparse = K.dot(K.T)/2*total_edges
K_outer_sparse = np.outer(K,K)/(total_edges)
Z_sparse = A_sparse - K_outer_sparse



#Z_sparse = Z_sparse + result_lambda_I_sparse
#print(Z_sparse)

n = Z_sparse.shape[0]  

x = np.random.rand(n)

x= x/np.linalg.norm(x)

num_iter = 250


Z_sparse_final = Z_sparse.A

for _ in range (num_iter):
    x_new = Z_sparse_final.dot(x)
    x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

    x = x_new

x_final = x

print(f"the leading eigen vector is : {x_final}")
print(f"the the resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")





identity_sparse = diags([np.ones((len(K)))], [0], shape=(len(K), len(K)))

# Multiply the sparse identity matrix by the scalar
result_lambda_I_sparse = (x_final.T.dot(Z_sparse_final).dot(x_final)) * identity_sparse

Z_sparse_final = Z_sparse - result_lambda_I_sparse
Z_sparse_final = Z_sparse_final.A

n = Z_sparse.shape[0]  

x = np.random.rand(n)

x= x/np.linalg.norm(x)

for _ in range (num_iter):
    x_new = Z_sparse_final.dot(x)
    x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

    x = x_new

x_final = x

print(f"the modified leading eigen vector is : {x_final}")
print(f"the modified  resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")

######## new part

cluster_one = x_final > 0
print(cluster_one)
# Assign clusters based on eigenvector signs
community_labels = (x_final > 0).astype(int)

# Step 5: Export the Graph for Gephi
#save_csrmatrix_Gephi_gexf_twocolors(A_sparse, "karate-network.gexf", community_labels)

#print("Graph saved with community labels for Gephi visualization!")


###new part:
import numpy as np
import snap

def compute_modularity(G, community_labels):
    # Get all nodes in the graph
    node_ids = [node.GetId() for node in G.Nodes()]
    num_nodes = len(node_ids)  # Number of nodes in the graph
    m = G.GetEdges()  # Total number of edges
    
    # Initialize modularity
    Q = 0.0
    
    # Ensure community_labels is the correct size
    if len(community_labels) != num_nodes:
        raise ValueError(f"Community labels length ({len(community_labels)}) does not match number of nodes ({num_nodes})")
    
    # Create the adjacency matrix A using a dictionary to handle arbitrary node IDs
    A = {i: {} for i in range(num_nodes)}  # Dictionary of dictionaries
    for edge in G.Edges():
        u, v = edge.GetSrcNId(), edge.GetDstNId()
        if u not in A:
            A[u] = {}
        if v not in A:
            A[v] = {}
        A[u][v] = A[v][u] = 1  # Since the graph is undirected
    
    # Compute the degree of each node
    degrees = np.array([G.GetNI(node_id).GetDeg() for node_id in node_ids])
    
    # Compute modularity
    for i in range(num_nodes):
        for j in range(num_nodes):
            # If nodes i and j are in the same community
            if community_labels[i] == community_labels[j]:
                Q += A[i].get(j, 0) - (degrees[i] * degrees[j]) / (2 * m)
    
    # Normalize by the total number of edges
    Q /= (2 * m)
    
    return Q

# Example usage:
#G = snap.LoadEdgeList(snap.PUNGraph, "zachary.net", 0, 1)  # Load the Zachary Karate Club network
#community_labels = (x > 0).astype(int)  # Assuming x is your leading eigenvector result from earlier

# Ensure that community_labels have the correct size
community_labels = community_labels[:G.GetNodes()]  # Adjust the size if necessary

modularity = compute_modularity(G, community_labels)
print(f"Modularity: {modularity}")






import sys
import snap
import numpy as np
sys.path.append("/courses/TSKS33/ht2024/common-functions")
from save_Gephi_gexf import saveGephi_gexf
from save_Gephi_gexf import saveGephi_gexf_twocolors
from gen_stochblock import gen_stoch_block_2comm
import snap_scipy as sp
from save_Gephi_gexf import save_csrmatrix_Gephi_gexf_twocolors
from scipy.sparse import diags
#G = snap.LoadEdgeList(snap.PUNGraph, "test-network.txt", 0, 1)

G = snap.LoadEdgeList(snap.PUNGraph, "karate-network.txt", 0, 1)


#G = snap.LoadEdgeList(snap.PUNGraph, "SB-small-network.txt", 0, 1)
#G = snap.LoadEdgeList(snap.PUNGraph, "SB-large-network.txt", 0, 1)

#snap.SaveMatlabSparseMtx(G, "test-network.mat")

A_sparse =sp.to_sparse_mat(G)
#print(A)

K = A_sparse.sum(axis=1).A.flatten()  # .A converts the result to a dense array

# Step 2: Create a sparse diagonal matrix from the degree vector
K_sparse = diags(K)  # Creates a sparse diagonal matrix with the degrees

# Now K_sparse is a sparse degree matrix, and you can use it efficiently
print(K_sparse)


print(f"{K} is the degree matrix")
m = K_sparse.sum(axis = 0)
print(f"twice the number of edges {m}")
print(f"the sparse representation of the degree matrix : {K_sparse} ")


#from s5 import solve_test_network

#x = solve_test_network()

total_edges = K_sparse.sum()
#csr_matrix
print(f"total edges : {total_edges}")

#K_outer_sparse = K.dot(K.T)/2*total_edges
K_outer_sparse = np.outer(K,K)/(total_edges)
Z_sparse = A_sparse - K_outer_sparse



#Z_sparse = Z_sparse + result_lambda_I_sparse
#print(Z_sparse)

n = Z_sparse.shape[0]  

x = np.random.rand(n)

x= x/np.linalg.norm(x)

num_iter = 250


Z_sparse_final = Z_sparse.A

for _ in range (num_iter):
    x_new = Z_sparse_final.dot(x)
    x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

    x = x_new

x_final = x

print(f"the leading eigen vector is : {x_final}")
print(f"the the resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")





identity_sparse = diags([np.ones((len(K)))], [0], shape=(len(K), len(K)))

# Multiply the sparse identity matrix by the scalar
result_lambda_I_sparse = (x_final.T.dot(Z_sparse_final).dot(x_final)) * identity_sparse

Z_sparse_final = Z_sparse - result_lambda_I_sparse
Z_sparse_final = Z_sparse_final.A

n = Z_sparse.shape[0] 


x = np.random.rand(n)

x= x/np.linalg.norm(x)

for _ in range (num_iter):
    x_new = Z_sparse_final.dot(x)
    x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

    x = x_new

x_final = x

print(f"the modified leading eigen vector is : {x_final}")
print(f"the modified  resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")

######## new part

cluster_one = x_final > 0
print(cluster_one)
# Assign clusters based on eigenvector signs
community_labels = (x_final > 0).astype(int)

# Step 5: Export the Graph for Gephi
save_csrmatrix_Gephi_gexf_twocolors(A_sparse, "karate-network.gexf", community_labels)

print("Graph saved with community labels for Gephi visualization!")


###new part:
import numpy as np
import snap

def compute_modularity(G, community_labels):
    # Get all nodes in the graph
    node_ids = [node.GetId() for node in G.Nodes()]
    num_nodes = len(node_ids)  # Number of nodes in the graph
    m = G.GetEdges()  # Total number of edges
    
    # Initialize modularity
    Q = 0.0
    
    # Ensure community_labels is the correct size
    if len(community_labels) != num_nodes:
        raise ValueError(f"Community labels length ({len(community_labels)}) does not match number of nodes ({num_nodes})")
    
    # Create the adjacency matrix A using a dictionary to handle arbitrary node IDs
    A = {i: {} for i in range(num_nodes)}  # Dictionary of dictionaries
    for edge in G.Edges():
        u, v = edge.GetSrcNId(), edge.GetDstNId()
        if u not in A:
            A[u] = {}
        if v not in A:
            A[v] = {}
        A[u][v] = A[v][u] = 1  # Since the graph is undirected
    
    # Compute the degree of each node
    degrees = np.array([G.GetNI(node_id).GetDeg() for node_id in node_ids])
    
    # Compute modularity
    for i in range(num_nodes):
        for j in range(num_nodes):
            # If nodes i and j are in the same community
            if community_labels[i] == community_labels[j]:
                Q += A[i].get(j, 0) - (degrees[i] * degrees[j]) / (2 * m)
    
    # Normalize by the total number of edges
    Q /= (2 * m)
    
    return Q

# Example usage:
#G = snap.LoadEdgeList(snap.PUNGraph, "zachary.net", 0, 1)  # Load the Zachary Karate Club network
#community_labels = (x > 0).astype(int)  # Assuming x is your leading eigenvector result from earlier

# Ensure that community_labels have the correct size
community_labels = community_labels[:G.GetNodes()]  # Adjust the size if necessary

modularity = compute_modularity(G, community_labels)
print(f"Modularity: {modularity}")





import sys
import snap
import numpy as np
sys.path.append("/courses/TSKS33/ht2024/common-functions")
from save_Gephi_gexf import saveGephi_gexf
from save_Gephi_gexf import saveGephi_gexf_twocolors
from gen_stochblock import gen_stoch_block_2comm
import snap_scipy as sp
from save_Gephi_gexf import save_csrmatrix_Gephi_gexf_twocolors
from scipy.sparse import diags
#G = snap.LoadEdgeList(snap.PUNGraph, "test-network.txt", 0, 1)

#G = snap.LoadEdgeList(snap.PUNGraph, "karate-network.txt", 0, 1)


G = snap.LoadEdgeList(snap.PUNGraph, "SB-small-network.txt", 0, 1)
#G = snap.LoadEdgeList(snap.PUNGraph, "SB-large-network.txt", 0, 1)

#snap.SaveMatlabSparseMtx(G, "test-network.mat")

A_sparse =sp.to_sparse_mat(G)
#print(A)

K = A_sparse.sum(axis=1).A.flatten()  # .A converts the result to a dense array

# Step 2: Create a sparse diagonal matrix from the degree vector
K_sparse = diags(K)  # Creates a sparse diagonal matrix with the degrees

# Now K_sparse is a sparse degree matrix, and you can use it efficiently
print(K_sparse)


print(f"{K} is the degree matrix")
m = K_sparse.sum(axis = 0)
print(f"twice the number of edges {m}")
print(f"the sparse representation of the degree matrix : {K_sparse} ")


#from s5 import solve_test_network

#x = solve_test_network()

total_edges = K_sparse.sum()
#csr_matrix
print(f"total edges : {total_edges}")

#K_outer_sparse = K.dot(K.T)/2*total_edges
K_outer_sparse = np.outer(K,K)/(total_edges)
Z_sparse = A_sparse - K_outer_sparse



#Z_sparse = Z_sparse + result_lambda_I_sparse
#print(Z_sparse)

n = Z_sparse.shape[0]  

x = np.random.rand(n)

x= x/np.linalg.norm(x)

num_iter = 250


Z_sparse_final = Z_sparse.A

for _ in range (num_iter):
    x_new = Z_sparse_final.dot(x)
    x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

    x = x_new

x_final = x

print(f"the leading eigen vector is : {x_final}")
print(f"the the resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")





#identity_sparse = diags([np.ones((len(K)))], [0], shape=(len(K), len(K)))

# Multiply the sparse identity matrix by the scalar
#result_lambda_I_sparse = (x_final.T.dot(Z_sparse_final).dot(x_final)) * identity_sparse

#Z_sparse_final = Z_sparse - result_lambda_I_sparse
#Z_sparse_final = Z_sparse_final.A

#n = Z_sparse.shape[0]  

#x = np.random.rand(n)

#x= x/np.linalg.norm(x)

#for _ in range (num_iter):
   # x_new = Z_sparse_final.dot(x)
 #   x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

   # x = x_new

#x_final = x

#print(f"the modified leading eigen vector is : {x_final}")
#print(f"the modified  resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")

######## new part

cluster_one = x_final > 0
print(cluster_one)
# Assign clusters based on eigenvector signs
community_labels = (x_final > 0).astype(int)

# Step 5: Export the Graph for Gephi
#save_csrmatrix_Gephi_gexf_twocolors(A_sparse, "SB-small-network.gexf", community_labels)

#print("Graph saved with community labels for Gephi visualization!")


###new part:
import numpy as np
import snap

def compute_modularity(G, community_labels):
    # Get all nodes in the graph
    node_ids = [node.GetId() for node in G.Nodes()]
    num_nodes = len(node_ids)  # Number of nodes in the graph
    m = G.GetEdges()  # Total number of edges
    
    # Initialize modularity
    Q = 0.0
    
    # Ensure community_labels is the correct size
    if len(community_labels) != num_nodes:
        raise ValueError(f"Community labels length ({len(community_labels)}) does not match number of nodes ({num_nodes})")
    
    # Create the adjacency matrix A using a dictionary to handle arbitrary node IDs
    A = {i: {} for i in range(num_nodes)}  # Dictionary of dictionaries
    for edge in G.Edges():
        u, v = edge.GetSrcNId(), edge.GetDstNId()
        if u not in A:
            A[u] = {}
        if v not in A:
            A[v] = {}
        A[u][v] = A[v][u] = 1  # Since the graph is undirected
    
    # Compute the degree of each node
    degrees = np.array([G.GetNI(node_id).GetDeg() for node_id in node_ids])
    
    # Compute modularity
    for i in range(num_nodes):
        for j in range(num_nodes):
            # If nodes i and j are in the same community
            if community_labels[i] == community_labels[j]:
                Q += A[i].get(j, 0) - (degrees[i] * degrees[j]) / (2 * m)
    
    # Normalize by the total number of edges
    Q /= (2 * m)
    
    return Q

# Example usage:
#G = snap.LoadEdgeList(snap.PUNGraph, "zachary.net", 0, 1)  # Load the Zachary Karate Club network
#community_labels = (x > 0).astype(int)  # Assuming x is your leading eigenvector result from earlier

# Ensure that community_labels have the correct size
community_labels = community_labels[:G.GetNodes()]  # Adjust the size if necessary

modularity = compute_modularity(G, community_labels)
print(f"Modularity: {modularity}")




import sys
import snap
import numpy as np
sys.path.append("/courses/TSKS33/ht2024/common-functions")
from save_Gephi_gexf import saveGephi_gexf
from save_Gephi_gexf import saveGephi_gexf_twocolors
from gen_stochblock import gen_stoch_block_2comm
import snap_scipy as sp
from save_Gephi_gexf import save_csrmatrix_Gephi_gexf_twocolors
from scipy.sparse import diags
#G = snap.LoadEdgeList(snap.PUNGraph, "test-network.txt", 0, 1)

#G = snap.LoadEdgeList(snap.PUNGraph, "karate-network.txt", 0, 1)


#G = snap.LoadEdgeList(snap.PUNGraph, "SB-small-network.txt", 0, 1)
G = snap.LoadEdgeList(snap.PUNGraph, "SB-large-network.txt", 0, 1)

#snap.SaveMatlabSparseMtx(G, "test-network.mat")

A_sparse =sp.to_sparse_mat(G)
#print(A)

#K = A_sparse.sum(axis=1).A.flatten()  # .A converts the result to a dense array
K = []

# Loop over each node and get its degree
for node in G.Nodes():
    K.append(node.GetDeg())
print(f"K is : {K}")
#K = np.array([node.GetDeg() for node in G.Nodes()]) 
#K = G.GetDegreee
# Step 2: Create a sparse diagonal matrix from the degree vector
K_sparse = diags(K)  # Creates a sparse diagonal matrix with the degrees

# Now K_sparse is a sparse degree matrix, and you can use it efficiently
#print(K_sparse)


#print(f"{K} is the degree matrix")
#m = K_sparse.sum(axis = 0)
#print(f"twice the number of edges {m}")
#print(f"the sparse representation of the degree matrix : {K_sparse} ")


#from s5 import solve_test_network

#x = solve_test_network()

#total_edges = K_sparse.sum()
total_edges = G.GetEdges()
#csr_matrix
print(f"total edges : {total_edges}")

#K_outer_sparse = K.dot(K.T)/2*total_edges
K_outer_sparse = np.outer(K,K)/(2*total_edges)
Z_sparse = A_sparse - K_outer_sparse



#Z_sparse = Z_sparse + result_lambda_I_sparse
#print(Z_sparse)

#n = Z_sparse.shape[0]  
n = G.GetNodes()
x = np.random.rand(n)

x= x/np.linalg.norm(x)

num_iter = 250

from scipy.sparse.linalg import eigs

eigenvalues, eigenvectors = eigs(Z_sparse, k =1)
print("Eigenvalues:", eigenvalues)
#print("Eigenvectors:\n", eigenvectors)

Z_sparse_final = Z_sparse.A

for _ in range (num_iter):
    x_new = Z_sparse_final.dot(x)
    x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

    x = x_new

x_final = x

print(f"the leading eigen vector is : {x_final}")
print(f"the the resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")





#identity_sparse = diags([np.ones((len(K)))], [0], shape=(len(K), len(K)))

# Multiply the sparse identity matrix by the scalar
#result_lambda_I_sparse = (x_final.T.dot(Z_sparse_final).dot(x_final)) * identity_sparse

#Z_sparse_final = Z_sparse - result_lambda_I_sparse
#Z_sparse_final = Z_sparse_final.A

#n = Z_sparse.shape[0]  

#x = np.random.rand(n)

#x= x/np.linalg.norm(x)

#for _ in range (num_iter):
   # x_new = Z_sparse_final.dot(x)
 #   x_new = x_new/np.linalg.norm(x_new)

    #if np.linalg.norm(x_new - x) < threshold:

        
        #break

   # x = x_new

#x_final = x

#print(f"the modified leading eigen vector is : {x_final}")
#print(f"the modified  resulting eigen value is : {x_final.T.dot(Z_sparse).dot(x_final)}")

######## new part

cluster_one = x_final > 0
#print(cluster_one)
# Assign clusters based on eigenvector signs
community_labels = (x_final > 0).astype(int)

# Step 5: Export the Graph for Gephi
#save_csrmatrix_Gephi_gexf_twocolors(A_sparse, "SB-large-network.gexf", community_labels)

#print("Graph saved with community labels for Gephi visualization!")


##last part : 
#eigenvalues, eigenvectors = np.linalg.eig(Z_sparse_final)




###new part:
import numpy as np
import snap

def compute_modularity(G, community_labels):
    # Get all nodes in the graph
    node_ids = [node.GetId() for node in G.Nodes()]
    num_nodes = len(node_ids)  # Number of nodes in the graph
    m = G.GetEdges()  # Total number of edges
    
    # Initialize modularity
    Q = 0.0
    
    # Ensure community_labels is the correct size
    if len(community_labels) != num_nodes:
        raise ValueError(f"Community labels length ({len(community_labels)}) does not match number of nodes ({num_nodes})")
    
    # Create the adjacency matrix A using a dictionary to handle arbitrary node IDs
    A = {i: {} for i in range(num_nodes)}  # Dictionary of dictionaries
    for edge in G.Edges():
        u, v = edge.GetSrcNId(), edge.GetDstNId()
        if u not in A:
            A[u] = {}
        if v not in A:
            A[v] = {}
        A[u][v] = A[v][u] = 1  # Since the graph is undirected
    
    # Compute the degree of each node
    degrees = np.array([G.GetNI(node_id).GetDeg() for node_id in node_ids])
    
    # Compute modularity
    for i in range(num_nodes):
        for j in range(num_nodes):
            # If nodes i and j are in the same community
            if community_labels[i] == community_labels[j]:
                Q += A[i].get(j, 0) - (degrees[i] * degrees[j]) / (2 * m)
    
    # Normalize by the total number of edges
    Q /= (2 * m)
    
    return Q

# Example usage:
#G = snap.LoadEdgeList(snap.PUNGraph, "zachary.net", 0, 1)  # Load the Zachary Karate Club network
#community_labels = (x > 0).astype(int)  # Assuming x is your leading eigenvector result from earlier

# Ensure that community_labels have the correct size
community_labels = community_labels[:G.GetNodes()]  # Adjust the size if necessary

#modularity = compute_modularity(G, community_labels)
#print(f"Modularity: {modularity}")
