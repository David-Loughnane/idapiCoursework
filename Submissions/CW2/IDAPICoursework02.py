import numpy as np
import IDAPICourseworkLibrary as cwLib

# Coursework 1 begins here

# Function to compute the prior distribution of the variable root from the data set
def calc_prior(the_data, root, no_states):
    num_obs = the_data[3]
    prior = np.zeros((no_states[root]), float)
    for state in the_data[4]:
        observation = state[root]
        prior[observation] += 1
    prior /= num_obs
    return prior


# Function to compute a CPT with parent node varP and child node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def calc_cond_prob(the_data, child, parent, no_states):
    link_matrix = np.zeros((no_states[child], no_states[parent]), float)
    for state in the_data[4]:
        parent_ob = state[parent]
        child_ob = state[child]
        link_matrix[child_ob][parent_ob] += 1
    row_sum = link_matrix.sum(axis=0)
    for row in link_matrix:
        for i in range(0, no_states[parent]):
            if row_sum[i] != 0:
                row[i] /= row_sum[i]
    return link_matrix


# Function to calculate the joint probability table of two variables in the data set
def calc_joint_prob(the_data, var_row, var_col, no_states):
    num_obs = the_data[3]
    joint_prob = np.zeros((no_states[var_row], no_states[var_col]), float)
    for state in the_data[4]:
        row_ob = state[var_row]
        col_ob = state[var_col]
        joint_prob[row_ob][col_ob] += 1
    joint_prob /= num_obs
    return joint_prob


# Function to convert a joint probability table to a conditional probability table
def convert_joint_2_cond(joint_prob):
    row_sum = joint_prob.sum(axis=0)
    no_states = joint_prob.shape[1]
    for row in joint_prob:
        for i in range(0, no_states):
            if row_sum[i] != 0:
                row[i] /= row_sum[i]
    return joint_prob


# Function to query a naive Bayesian network
def query(the_query, naive_bayes):
    for root in range(0, naive_bayes[1]):
        root_pdf = calc_prior(naive_bayes, root, naive_bayes[2])
        posterior_prob = root_pdf
        for node in range(naive_bayes[1], naive_bayes[0]):
            instan_value = the_query[node - naive_bayes[1]]
            cond_prob_node = calc_cond_prob(naive_bayes, node, root, naive_bayes[2])
            cond_prob_instan = cond_prob_node[instan_value]
            posterior_prob *= cond_prob_instan.transpose()
        norm_constant = posterior_prob.sum()
        posterior_prob /= norm_constant
    return posterior_prob

# End of Coursework 1


# Coursework 2 begins here

# Function to calculate the mutual information from the joint probability table of two variables
def calc_kullback_lieb(joint_prob):
    no_states = joint_prob.shape

    # marginalisation of joint_prob
    prob_a = np.zeros(no_states[0])
    for i in range(no_states[0]):
        for prob in joint_prob[i]:
            prob_a[i] = prob_a[i] + prob
    prob_b = np.zeros(no_states[1])
    for j in range(no_states[1]):
        for i in range(no_states[0]):
            prob_b[j] = prob_b[j] + joint_prob[i][j]

    mutual_info = 0.0
    for i in range(no_states[0]):
        for j in range(no_states[1]):
            if not ((prob_a[i] == 0.0) or (prob_b[j] == 0.0) or (joint_prob[i][j] == 0.0)):
                mutual_info = mutual_info + (joint_prob[i][j] * np.log2((joint_prob[i][j] / (prob_a[i] * prob_b[j]))))
    return mutual_info


# Function to construct a dependency matrix for all the variables
def calc_depen_matrix(the_data):
    depen_matrix = np.zeros((the_data[0], the_data[0]))

    for i in range(the_data[0]):
        for j in range(i, the_data[0]):
            dependency = calc_kullback_lieb(calc_joint_prob(the_data, i, j, the_data[2]))
            depen_matrix[i][j] = dependency
            depen_matrix[j][i] = dependency
    return depen_matrix


# Function to compute an ordered list of dependencies 
def create_depen_list(depen_matrix):
    depen_list = []
    matrix_size = len(depen_matrix)
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            if i != j:
                depen_list.append([depen_matrix[i][j], i, j])
    depen_list.sort(reverse=True)
    return depen_list


# Function implementing the spanning tree algorithm
def generate_span_tree(depen_list, num_nodes):
    spanning_tree = {}
    for i in range(len(depen_list)):
        if depen_list[i][1] not in spanning_tree.keys():
            spanning_tree[depen_list[i][1]] = list()
        #IF THERE IS NO CYCLE:
        spanning_tree[depen_list[i][1]].append(depen_list[i][2])
        if depen_list[i][2] not in spanning_tree.keys():
            spanning_tree[depen_list[i][2]] = list()
        #IF THERE IS NO CYCLE:
        spanning_tree[depen_list[i][2]].append(depen_list[i][1])
    return spanning_tree

# End of coursework 2



# COURSEWORK 2 MAIN
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