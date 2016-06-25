import numpy as np
import IDAPICourseworkLibrary as cwLib

# Function to compute the prior distribution of the variable root from the data set
def calc_prior(the_data, root, no_states):
    num_obs = the_data[3]
    prior = np.zeros((no_states[root]), float)
    for state in the_data[4]:
        observation = state[root]
        prior[observation] += 1
    prior /= num_obs
    return prior


# Function to compute a CPT with parent node and child node from the data array
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
            cond_prob_instan =  cond_prob_node[instan_value]
            posterior_prob *= cond_prob_instan.transpose()
        norm_constant = posterior_prob.sum()
        posterior_prob /= norm_constant
    return posterior_prob



neurones = cwLib.readfile("Neurones.txt")
print 'RAW DATA'
print neurones
print '\n'


# TASK 1.1 Prior Probability of Root Node
prior_prob = calc_prior(neurones, 0, neurones[2])
print 'PRIOR PROBABILITY'
print prior_prob
print '\n'


# TASK 1.2 Conditional Probability of Node 2 on Root Node
cond_prob = calc_cond_prob(neurones, 2, 0, neurones[2])
print 'CONDITIONAL PROBABILITY'
print cond_prob
print '\n'


# TASK 1.3 Joint Probability of Node 2 and Root Node
joint_prob = calc_joint_prob(neurones, 2, 0, neurones[2])
print 'JOINT PROBABILITY'
print joint_prob
print '\n'


# TASK 1.4 Conditional Probability of Node 2 on Root Node from their Joint Probability
cond_prob = convert_joint_2_cond(joint_prob)
print 'CONDITIONAL PROBABILITY (FROM JOINT)'
print cond_prob
print '\n'


# TASK 1.5 Posterior Probabilities given evidence vector
post_prob = query([4, 0, 0, 0, 5], neurones)
print 'POSTERIOR PROBABILITY'
print post_prob
print '\n'

post_prob = query([6, 5, 2, 5, 5], neurones)
print 'POSTERIOR PROBABILITY'
print post_prob
print '\n'