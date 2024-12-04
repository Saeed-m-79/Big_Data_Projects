#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 4, load data


"""
import random
import snap
# from gen_data import genmod10star
from gen_data import genLiveJournal

# -- load 10-star --
# G, h = genmod10star()
S = 10 ** 4
# -- load LiveJournal --
G, h = genLiveJournal()
# print(G)
# print(G.GetNodes())
# print(G.GetEdges())

###Task 1 - Uniform Sampling:

Ids = []
i = 1
for NI in G.Nodes():
    # print(f" the {i}'s ID is : {NI.GetId()}")
    Ids.append(NI.GetId())
    i += 1
i = 0
degrees = []
for NI in G.Nodes():
    # print(f" the {i}'s Degree is : {NI.GetDeg()}")
    degrees.append(NI.GetDeg())
    i += 1

# print(degrees)


summation = 0
attri_exact = []
###############################################################################
# Precompute degrees for all nodes
node_degrees = {NI.GetId(): NI.GetOutDeg() for NI in G.Nodes()}

# Initialize summation and attribute list
attri_exact = []

# Iterate over all node IDs
for Id in Ids:
    # Get the neighbors of the current node
    node_NI = G.GetNI(Id)
    neighbours = [node_NI.GetOutNId(i) for i in range(node_NI.GetOutDeg())]

    # Compute the summation of the inverse degrees of neighbors
    summation = sum(1 / node_degrees[neighbour] for neighbour in neighbours)

    # Append the weighted attribute value
    attri_exact.append(summation * h[Id])

# Compute and print the final result
print(f"Exact value for random neighbour of a random node: {sum(attri_exact) / len(attri_exact)}")

# print("Node Attributes:", h)

attributs = []
for key in h:
    # print(f"Node ID: {key}, Attribute: {h[key]}")
    attributs.append(h[key])

print(f" the exact <x> value for uniform sampling  is: {sum(attributs) / len(Ids)} ")

############################################################################33


attr_neigh_NI = []
for Id in Ids:  ## 4 to be replaced with S(predefinded up)
    # Get all neighbors
    node_NI = G.GetNI(Id)
    neighbours = [node_NI.GetOutNId(i) for i in range(node_NI.GetOutDeg())]
    # sample_rand_neigh = random.sample(neighbours, 1)[0]
    for neighbour in neighbours:
        attr_neigh_NI.append(h[neighbour])

print(f"the exact <x> for uniform random walk of all the node is : {sum(attr_neigh_NI) / len(attr_neigh_NI)}")
# print(f"the true value and the epected value were calculated before")

attributs = []
for key in h:
    # print(f"Node ID: {key}, Attribute: {h[key]}")
    attributs.append(h[key])

print(f" the exact <x> value for metropolis hasting is: {sum(attributs) / len(Ids)} ")

##end of Task 1


# Task 2
for l in range(5):
    attr_uni_sam = []
    for j in range(S):
        samples_uni_sam = random.sample(Ids, 1)[0]  ### to be changed for examination
        # print(f"Ids of the samples taken : {samples_uni_sam}")

        # getting the attributes of the corrisponding nodes

        attr_uni_sam.append(h[samples_uni_sam])

    # print(f"the attributes are : {attr_uni_sam}")

    print(f"estimtion (<x^>) for {S} samples : {sum(attr_uni_sam) / len(attr_uni_sam)}")

# End of Task 2 Test

### end of task 2

# Task 3 - random_neighbours of random sampling:

for l in range(5):
    attr_neigh_NI = []
    for j in range(S):  ## 4 to be replaced with S(predefinded up)
        # Get all neighbors
        samples_node_NI = G.GetNI(random.sample(Ids, 1)[0])
        neighbours = [samples_node_NI.GetOutNId(i) for i in range(samples_node_NI.GetOutDeg())]
        sample_rand_neigh = random.sample(neighbours, 1)[0]
        attr_neigh_NI.append(h[sample_rand_neigh])

    print(f"the estimation for random neighbour of a random node is : {sum(attr_neigh_NI) / len(attr_neigh_NI)}")
    # print(f"the true value and the epected value were calculated before")

###End of Task 3

# Task 4 - Uniform Random-Walk Sampling:
for l in range(5):
    attr_neigh_NI = []
    # picking a random initial sample :
    random_starting_point = random.sample(Ids, 1)[0]

    for j in range(10000):  ## 4 to be replaced with S (predefinded up)
        # Get all neighbors
        samples_node_NI = G.GetNI(random_starting_point)
        neighbours = [samples_node_NI.GetOutNId(i) for i in range(samples_node_NI.GetOutDeg())]
        # print(neighbours)
        sample_rand_neigh = random.sample(neighbours, 1)[0]
        random_starting_point = sample_rand_neigh
        # print(random_starting_point)
        # now we have reached the steady-state

    # now we use the last point to take S nodes:
    attr_neigh_NI = []
    for j in range(S):  ## 10 to be replaced with S (predefinded up)
        # Get all neighbors
        samples_node_NI = G.GetNI(random_starting_point)
        neighbours = [samples_node_NI.GetOutNId(i) for i in range(samples_node_NI.GetOutDeg())]
        # print(neighbours)
        sample_rand_neigh = random.sample(neighbours, 1)[0]

        random_starting_point = sample_rand_neigh
        attr_neigh_NI.append(h[sample_rand_neigh])
        # print(random_starting_point)
        # now we have reached the steady-state

    print(f"the estimation of <x> for uniform random walk is : {sum(attr_neigh_NI) / len(attr_neigh_NI)}")

##End of Task 4

# Task 5 - Metropolis-Hasting Random Walk:


for l in range(5):

    # picking a random initial sample :
    random_starting_point = random.sample(Ids, 1)[0]
    # again the burning phase:
    for j in range(10000):  ## 4 to be replaced with S (predefinded up)
        # Get all neighbors
        samples_node_NI = G.GetNI(random_starting_point)
        neighbours = [samples_node_NI.GetOutNId(i) for i in range(samples_node_NI.GetOutDeg())]
        # print(neighbours)
        sample_rand_neigh = random.sample(neighbours, 1)[0]
        random_starting_point = sample_rand_neigh
        # print(random_starting_point)
        # now we have reached the steady-state

    # now we use the last point to take S nodes:

    attr_neigh_NI = []
    for j in range(S):  ## 10 to be replaced with S (predefinded up)
        # Get all neighbors
        p = random.random()
        samples_node_NI = G.GetNI(random_starting_point)
        kn_prime = samples_node_NI.GetDeg()
        neighbours = [samples_node_NI.GetOutNId(i) for i in range(samples_node_NI.GetOutDeg())]
        # print(f"the neighbours for the node  {random_starting_point} are : {neighbours}")
        # print(neighbours)
        sample_rand_neigh = random.sample(neighbours, 1)[0]
        neighbour_node_NI = G.GetNI(sample_rand_neigh)
        kn = neighbour_node_NI.GetDeg()
        # print(f"the chosen neighbour is : {sample_rand_neigh}")
        if p > kn_prime / kn:
            # print('no')
            # print(f"{h[sample_rand_neigh]} <= {h[random_starting_point]}")
            random_starting_point = random_starting_point
            attr_neigh_NI.append(h[random_starting_point])
        if p <= kn_prime / kn:
            # print('yes')
            # print(f"{h[sample_rand_neigh]} > {h[random_starting_point]}")
            random_starting_point = sample_rand_neigh
            attr_neigh_NI.append(h[random_starting_point])
        # random_starting_point = sample_rand_neigh
        # print(random_starting_point)
        # now we have reached the steady-state

    print(f"the estimation of <x> for M-H method is : {sum(attr_neigh_NI) / len(attr_neigh_NI)}")

    # print(h.values())




