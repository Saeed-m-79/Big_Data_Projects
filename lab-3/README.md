This is a project in the course 'Big Data and Complex netwroks' at Linkoping University of Sweden where I am studying as an exchange student.

Summary of the Task The task is to implement a few centrality metrics and experiment with them on sub�networks of the English Wikipedia project. Nodes correspond to Wikipedia pages, and edges correspond to links between the pages.

Task 1 Calculate the in- and out-degrees of all articles. Show the results in two lists: one list that shows the in- and out-degrees of the five nodes with the highest in-degree, and one list shows the in- and out-degrees of the five nodes with highest out-degree. After computing the centrality scores, normalize them so that they sum to one.

Task 2 Calculate the hub and authority centrality of all articles. Display the result in the same way as in the previous task. After computing the centrality scores, normalize them so that they sum to one.

Task 3 Calculate the eigenvector centrality of all articles. Use the eigenvector corresponding to the largest eigenvalue. Display the result in the same way as in the previous task.

After computing the centrality scores, normalize them so that they sum to one. Task 4 Calculate the Katz Centrality of each article, using 𝛼 = 0.85 · 1/|𝜆max| , where 𝜆max is the largest eigenvalue of the adjacency matrix. List the top five articles and their Katz score. After computing the centrality scores, normalize them so that they sum to one.

Task 5 Calculate the Google PageRank score of each article in your network. Use the version of PageRank explained in the course material, and compute the PageRank scores by using the closed-form solution. Use the parameter value 𝛼 = 0.85. List the top five articles and their PageRank scores. Also try some other values of 𝛼 and comment on the result. After computing the centrality scores, normalize them so that they sum to one. Task 6 Implement the iterative version of PageRank. For the top-three articles found in Task 5, plot how the PageRank score evolves for 1, 2, 3, ..., 100 iterations. In the same plot, show also the “exact” closed-form solution from Task 5. After computing the centrality scores, normalize them so that they sum to one.
