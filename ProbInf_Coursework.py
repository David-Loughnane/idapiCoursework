import IDAPICourseworkSkeleton as cwSkel
import IDAPICourseworkLibrary as cwLib
import numpy as np

# COURSEWORK 1
'''
neurones = cwLib.readfile("Neurones.txt")
print 'RAW DATA'
print neurones
print '\n'

# TASK 1.1 Prior Probability of Root Node
prior_prob = cwSkel.calc_prior(neurones, 0, neurones[2])
print 'PRIOR PROBABILITY'
print prior_prob
print '\n'

# TASK 1.2 Conditional Probability of Node 2 on Root Node
cond_prob = cwSkel.calc_cond_prob(neurones, 2, 0, neurones[2])
print 'CONDITIONAL PROBABILITY'
print cond_prob
print '\n'


# TASK 1.3 Joint Probability of Node 2 and Root Node
joint_prob = cwSkel.calc_joint_prob(neurones, 2, 0, neurones[2])
print 'JOINT PROBABILITY'
print joint_prob
print '\n'


# TASK 1.4 Conditional Probability of Node 2 on Root Node from their Joint Probability
cond_prob = cwSkel.convert_joint_2_cond(joint_prob)
print 'CONDITIONAL PROBABILITY (FROM JOINT)'
print cond_prob
print '\n'


# TASK 1.5 Posterior Probabilities given evidence vector
post_prob = cwSkel.query([4, 0, 0, 0, 5], neurones)
print 'POSTERIOR PROBABILITY'
print post_prob
print '\n'

post_prob = cwSkel.query([6, 5, 2, 5, 5], neurones)
print 'POSTERIOR PROBABILITY'
print post_prob
print '\n'
'''

# COURSEWORK 2
'''
hepC = cwLib.readfile("HepatitisC.txt")
print 'RAW DATA'
print hepC
print '\n'

# TASK 2.1 Calculate Kullback Leibler Divergence between 2 variables
joint_prob = cwSkel.calc_joint_prob(hepC, 2, 7, hepC[2])
print 'JOINT PROBABILITY'
print joint_prob
print '\n'
mutual_entropy = cwSkel.calc_kullback_lieb(joint_prob)
print 'KULLBACK LIEBLER DIVERGENCE (MUTUAL ENTROPY)'
print mutual_entropy
print '\n'

# TASK 2.2 Calculate matrix of pairwise dependencies between variables - "Depedency Matrix"
depen_matrix = cwSkel.calc_depen_matrix(hepC)
print 'DEPENDENCY MATRIX'
print depen_matrix
print '\n'

# TASK 2.3 Create list of pairwise dependencies between variables from Dependency Matrix
#   [dependency, node1_id, nod2_id] ordered by magnitude of dependency
depen_list = cwSkel.create_depen_list(depen_matrix)
print 'DEPENDENCY LIST'
print depen_list
print '\n'

# TASK 2.4 Automatically generate the max weight spanning tree
spanning_tree = cwSkel.generate_span_tree(depen_list, hepC[0])
print 'MAX WEIGHT SPANNING TREE'
print spanning_tree
print '\n'
'''

# COURSEWORK 3
'''
hepC = cwLib.readfile("HepatitisC.txt")

print 'RAW DATA'
print hepC
print '\n'


# TASK 3.1 Calculate the link matrix for a variable with two parents
cond_prob = cwSkel.calc_cond_prob_2_parents(hepC, 2, 0, 1, hepC[2])
print 'CONDITIONAL PROBABILITY'
print cond_prob
print '\n'

# TASK 3.2 Returns a Bayesian Net data structure as a list of edges, and link matrices
arc_list, cpt_list = cwSkel.hep_network(hepC, hepC[2])

print 'BAYESIAN NETWORK\n'
print 'ARC LIST'
print arc_list
print '\n'
print 'CPT LIST'
print cpt_list
print '\n'


# TASK 3.3 Calculates the minimum description length size
# m-1 for each prior, (n-1)*(m for each parent) for each conditional
mdl_size = cwSkel.MDL_Size(arc_list, cpt_list, hepC[3], hepC[2])
print 'MDL SIZE'
print mdl_size
print '\n'


# TASK 3.4 Calculates the joint probability of a data point
joint_prob = cwSkel.calc_joint_prob_of_point([0, 8,	0,	0,	3,	6, 11,	5,	0], arc_list, cpt_list)
print 'JOINT PROBABILITY OF A DATA POINT'
print joint_prob
print '\n'


# TASK 3.5 Calculates the minimum description length accuracy
mdl_accuracy = cwSkel.MDL_accuracy(hepC, arc_list, cpt_list)
print 'MDL ACCURACY'
print mdl_accuracy
print '\n'


mdl_score = cwSkel.MDL_score(hepC, arc_list, cpt_list, hepC[3], hepC[2])
print 'MDL SCORE'
print mdl_score
print '\n'


# TASK 3.6 Find best scoring network by removing one arc from spanning tree
min_score = cwSkel.lowest_score(hepC, arc_list, cpt_list, hepC[3], hepC[2])
print 'BEST TREE MINUS 1 ARC'
print min_score
print '\n'
'''

# COURSEWORK 4

hepC = cwLib.readfile("HepatitisC.txt")
hepC_input = np.array(hepC[4])

# TASK 4.1 Calculate mean
the_mean = cwSkel.Mean(hepC_input)
print 'HepC Mean \n'
print the_mean
print '\n'


# TASK 4.2 Calculate Covariance
covar_matrix = cwSkel.Covariance(hepC_input)
print 'HepC Covariance \n'
print covar_matrix
print '\n'


# TASK 4.3 Create Eigenface images
eigenface_basis = cwLib.ReadEigenfaceBasis()
cwSkel.CreateEigenfaceFiles(eigenface_basis)

# TASK 4.4 Project face onto principal component basis
all_images = np.array(cwLib.ReadImages())
mean_all_images = np.array(cwSkel.Mean(all_images))
input_image = np.array(cwLib.ReadOneImage('c.pgm'))

project_face = cwSkel.ProjectFace(eigenface_basis, mean_all_images, input_image)
print 'c.pgm Component Magnitudes \n'
print project_face
print '\n'

# TASK 4.5 Generate and save image files of reconstruction from mean to full reconstruction
cwSkel.CreatePartialReconstructions(eigenface_basis, mean_all_images, project_face)

# TASK 4.6 Function to perform PCA on data set (KL method), return orthonormal basis
