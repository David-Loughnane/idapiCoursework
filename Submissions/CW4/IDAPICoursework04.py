import numpy as np
from math import log
import scipy as sc
import IDAPICourseworkLibrary as cwLib

'''
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


# Function to compute a CPT with one parent
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
            prob_b[j] += + joint_prob[i][j]

    mutual_info = 0.0
    for i in range(no_states[0]):
        for j in range(no_states[1]):
            if not ((prob_a[i] == 0.0) or (prob_b[j] == 0.0) or (joint_prob[i][j] == 0.0)):
                mutual_info += + (joint_prob[i][j] * np.log2((joint_prob[i][j] / (prob_a[i] * prob_b[j]))))
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
        # IF THERE IS NO CYCLE:
        spanning_tree[depen_list[i][1]].append(depen_list[i][2])
        if depen_list[i][2] not in spanning_tree.keys():
            spanning_tree[depen_list[i][2]] = list()
        # IF THERE IS NO CYCLE:
        spanning_tree[depen_list[i][2]].append(depen_list[i][1])
    return spanning_tree


# End of coursework 2


# Coursework 3 begins here

# Function to compute a CPT with two parents
def calc_cond_prob_2_parents(the_data, child, parent1, parent2, no_states):
    link_matrix = np.zeros((no_states[child], no_states[parent1], no_states[parent2]), float)
    for state in the_data[4]:
        link_matrix[state[child]][state[parent1]][state[parent2]] += 1
    row_sum = link_matrix.sum(axis=0)
    for row in link_matrix:
        for i in range(0, no_states[parent1]):
            for j in range(0, no_states[parent2]):
                if row_sum[i][j] != 0:
                    row[i][j] /= row_sum[i][j]
            print ' '
    return link_matrix


# Definition of a Bayesian Network
def ex_bayesian_net(the_data, no_states):
    arc_list = [[0], [1], [2, 0], [3, 2, 1], [4, 3], [5, 3]]
    cpt0 = calc_prior(the_data, 0, no_states)
    cpt1 = calc_prior(the_data, 1, no_states)
    cpt2 = calc_cond_prob(the_data, 2, 0, no_states)
    cpt3 = calc_cond_prob_2_parents(the_data, 3, 2, 1, no_states)
    cpt4 = calc_cond_prob(the_data, 4, 3, no_states)
    cpt5 = calc_cond_prob(the_data, 5, 3, no_states)
    cpt_list = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arc_list, cpt_list


# Definition of a Hepatitis Network
def hep_network(the_data, no_states):
    arc_list = [[0], [1], [3, 4], [5, 4], [4, 1], [6, 1], [7, 1, 0], [8, 7], [2, 0]]
    cpt0 = calc_prior(the_data, 0, no_states)
    cpt1 = calc_prior(the_data, 1, no_states)
    cpt2 = calc_cond_prob(the_data, 3, 4, no_states)
    cpt3 = calc_cond_prob(the_data, 5, 4, no_states)
    cpt4 = calc_cond_prob(the_data, 4, 1, no_states)
    cpt5 = calc_cond_prob(the_data, 6, 1, no_states)
    cpt6 = calc_cond_prob_2_parents(the_data, 7, 1, 0, no_states)
    cpt7 = calc_cond_prob(the_data, 8, 7, no_states)
    cpt8 = calc_cond_prob(the_data, 2, 0, no_states)
    cpt_list = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7, cpt8]
    return arc_list, cpt_list


# Function to calculate the MDL size of a Bayesian Network
def MDL_Size(arc_list, cpt_list, no_observations, no_states):
    mdl_size = 0.0
    for cpt in cpt_list:
        if len(cpt.shape) == 1:
            mdl_size = mdl_size + cpt.shape[0] - 1
        elif len(cpt.shape) == 2:
            mdl_size += (cpt.shape[0] - 1) * (cpt.shape[1])
        elif len(cpt.shape) == 3:
            mdl_size += (cpt.shape[0] - 1) * (cpt.shape[1]) * (cpt.shape[2])
    mdl_size *= (log(no_observations, 2)/2)
    return mdl_size


# Function to calculate the joint probability of a single data point in a Network
def calc_joint_prob_of_point(data_point, arc_list, cpt_list):
    joint_prob = 1.0
    for i in range(len(arc_list)):
        if len(arc_list[i]) == 1:
            joint_prob *= cpt_list[i][data_point[i]]
        elif len(arc_list[i]) == 2:
            joint_prob *= cpt_list[i][data_point[arc_list[i][0]]][data_point[arc_list[i][1]]]
        elif len(arc_list[i]) == 3:
            joint_prob *= cpt_list[i][data_point[arc_list[i][0]]][data_point[arc_list[i][1]]][
                data_point[arc_list[i][2]]]
    return joint_prob


# Function to calculate the MDLAccuracy from a data set
def MDL_accuracy(the_data, arc_list, cpt_list):
    mdl_accuracy = 0
    for i in range(0, len(arc_list)):
        if len(arc_list[i]) == 1:
            for j in range(0, the_data[2][arc_list[i][0]]):
                num_occurrences = 0
                for observation in the_data[4]:
                    if observation[arc_list[i][0]] == j:
                        num_occurrences += 1
                if num_occurrences != 0:
                    mdl_accuracy += num_occurrences * log(cpt_list[i][j], 2)

        if len(arc_list[i]) == 2:
            for j in range(0, the_data[2][arc_list[i][0]]):
                for k in range(0, the_data[2][arc_list[i][1]]):
                    num_occurrences = 0
                    for observation in the_data[4]:
                        if (observation[arc_list[i][0]] == j) and (observation[arc_list[i][1]] == k):
                            num_occurrences += 1
                    if num_occurrences != 0:
                        mdl_accuracy += num_occurrences * log(cpt_list[i][j][k], 2)

        if len(arc_list[i]) == 3:
            for j in range(0, the_data[2][arc_list[i][0]]):
                for k in range(0, the_data[2][arc_list[i][1]]):
                    for l in range(0, the_data[2][arc_list[i][2]]):
                        num_occurrences = 0
                        for observation in the_data[4]:
                            if (observation[arc_list[i][0]] == j) and (observation[arc_list[i][1]] == k) and (
                                        observation[arc_list[i][2]] == l):
                                num_occurrences += 1
                        if num_occurrences != 0:
                            mdl_accuracy += num_occurrences * log(cpt_list[i][j][k][l], 2)
    return mdl_accuracy


def MDL_score(the_data, arc_list, cpt_list, no_observations, no_states):
    mdl_size = MDL_Size(arc_list, cpt_list, no_observations, no_states)
    mdl_accuracy = MDL_accuracy(the_data, arc_list, cpt_list)
    mdl_score = mdl_size - mdl_accuracy
    return mdl_score


def lowest_score(the_data, arc_list, cpt_list, no_observations, no_states):
    min_score = MDL_score(the_data, arc_list, cpt_list, no_observations, no_states)
    for i in range(0, len(arc_list)):
        if len(arc_list[i]) == 2:
            temp_arc = arc_list[i]
            temp_cpt = cpt_list[i]
            arc_list[i] = [arc_list[i][0]]
            cpt_list[i] = calc_prior(the_data, arc_list[i][0], no_states)
            next_score = MDL_score(the_data, arc_list, cpt_list, no_observations, no_states)
            if next_score < min_score:
                min_score = next_score
            arc_list[i] = temp_arc
            cpt_list[i] = temp_cpt

        if len(arc_list[i]) == 3:
            temp_arc = arc_list[i]
            temp_cpt = cpt_list[i]
            arc_list[i] = [arc_list[i][0], arc_list[i][1]]
            cpt_list[i] = calc_cond_prob(the_data, arc_list[i][0], arc_list[i][1], no_states)
            next_score = MDL_score(the_data, arc_list, cpt_list, no_observations, no_states)
            if next_score < min_score:
                min_score = next_score
            arc_list[i] = temp_arc
            cpt_list[i] = temp_cpt

            temp_arc = arc_list[i]
            temp_cpt = cpt_list[i]
            arc_list[i] = [arc_list[i][0], arc_list[i][2]]
            cpt_list[i] = calc_cond_prob(the_data, arc_list[i][0], arc_list[i][1], no_states)
            next_score = MDL_score(the_data, arc_list, cpt_list, no_observations, no_states)
            if next_score < min_score:
                min_score = next_score
            arc_list[i] = temp_arc
            cpt_list[i] = temp_cpt
    return min_score

# End of coursework 3
'''


# Coursework 4 begins here
def Mean(input_data):
    float_data = input_data.astype(float)
    obs_count = float_data.shape[0]
    mean_array = np.zeros(float_data.shape[1])

    for array in float_data:
        mean_array += array

    mean_array /= obs_count
    return mean_array


def Covariance(input_data):
    input_data = input_data.astype(float)
    obs_count = input_data.shape[0]
    variable_count = input_data.shape[1]
    mean_array = Mean(input_data)
    covar = np.zeros((variable_count, variable_count), float)

    for x in range(0, variable_count):
        for y in range(0, variable_count):
            for i in range(0, obs_count):
                covar[x][y] += (input_data[i][x] - mean_array[x]) * (input_data[i][y] - mean_array[y])

    covar /= obs_count
    return covar


def CreateEigenfaceFiles(eigenface_basis):
    for i in range(0, 10):
        filename = 'PrincipalComponent' + str(i + 1) + '.jpg'
        cwLib.SaveEigenface(eigenface_basis[i], filename)


def ProjectFace(basis, mean, face_image):
    zero_mean_image = face_image - mean
    projection = np.dot(zero_mean_image, np.transpose(basis))
    return projection


def CreatePartialReconstructions(basis, mean, componentMags):
    cwLib.SaveEigenface(mean, "Reconstructed_Image_0" + ".jpg")
    for i in range(0, len(componentMags)):
        reconstructed_image = np.add(np.dot(np.transpose(basis[0:i]), componentMags[0:i]), mean)
        cwLib.SaveEigenface(reconstructed_image, "Reconstructed_Image_" + str(i + 1) + ".jpg")

'''
def PrincipalComponents(input_data):
    ortho_phi = []
    mean_data = Mean(input_data)

    U = input_data - mean_data
    Ut = np.transpose(U)
    UUt = np.dot(U, Ut)

    eigen_values, eigen_vectors = linalg.eig(UUt)
    Ut_eig = np.dot(Ut, eigen_vectors).transpose()

    for i in range(0, len(Ut_eig)):
        magnitude = np.math.sqrt(np.dot(Ut_eig[i], np.transpose(Ut_eig[i])))
        Ut_eig[i] /= magnitude



    return array(ortho_phi)
'''

# MAIN PROGRAM


# COURSEWORK 4

hepC = cwLib.readfile("HepatitisC.txt")
hepC_input = np.array(hepC[4])

# TASK 4.1 Calculate mean
the_mean = Mean(hepC_input)
print 'HepC Mean \n'
print the_mean
print '\n'


# TASK 4.2 Calculate Covariance
covar_matrix = Covariance(hepC_input)
print 'HepC Covariance \n'
print covar_matrix
print '\n'


# TASK 4.3 Create Eigenface images
eigenface_basis = cwLib.ReadEigenfaceBasis()
CreateEigenfaceFiles(eigenface_basis)

# TASK 4.4 Project face onto principal component basis
all_images = np.array(cwLib.ReadImages())
mean_all_images = np.array(Mean(all_images))
input_image = np.array(cwLib.ReadOneImage('c.pgm'))

project_face = ProjectFace(eigenface_basis, mean_all_images, input_image)
print 'c.pgm Component Magnitudes \n'
print project_face
print '\n'

# TASK 4.5 Generate and save image files of reconstruction from mean to full reconstruction
CreatePartialReconstructions(eigenface_basis, mean_all_images, project_face)

# TASK 4.6 Function to perform PCA on data set (KL method), return orthonormal basis