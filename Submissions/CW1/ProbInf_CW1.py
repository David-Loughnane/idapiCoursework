import IDAPICourseworkSkeleton as cwSkel
import IDAPICourseworkLibrary as cwLib

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

