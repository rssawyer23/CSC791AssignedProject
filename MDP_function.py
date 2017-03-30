import collections
import numpy as np
import pandas
import mdptoolbox, mdptoolbox.example
import argparse


def generate_MDP_input2(original_data, features):

    students_variables = ['student', 'priorTutorAction', 'reward']

    # generate distinct state based on feature
    #original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    original_data['state'] = original_data[features].apply(tuple, axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    # quantify actions
    distinct_acts = list(data['priorTutorAction'].unique())
    Nx = len(distinct_acts)
    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1

    # initialize state transition table, expected reward table, starting state table
    distinct_states = list(data['state'].unique())
    Ns = len(distinct_states)
    start_states = np.zeros(Ns)
    A = np.zeros((Nx, Ns, Ns))
    expectR = np.zeros((Nx, Ns, Ns))

    # update table values episode by episode
    # each episode is a student data set
    student_list = list(data['student'].unique())
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        for i in range(1, len(row_list)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        # generate expected reward
        # it has the warning, ignore it
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act][l][l] = 1
        A[act] = np.divide(A[act].transpose(), np.sum(A[act], axis=1))
        A[act] = A[act].transpose()

    return [start_states, A, expectR, distinct_acts, distinct_states]


def calcuate_ECR(start_states, expectV):
    ECR_value = start_states.dot(np.array(expectV))
    return ECR_value


def output_policy(distinct_acts, distinct_states, vi):
    Ns = len(distinct_states)
    print('Policy: ')
    print('state -> action, value-function')
    for s in range(Ns):
        print str(distinct_states[s]) + " -> " + str(distinct_acts[vi.policy[s]]) + ", " + str(vi.V[s])


def induce_policy_MDP2(original_data, selected_features):

    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    # output policy
    output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    print 'ECR value: %.5f' % ECR_value
    return ECR_value


# Function implemented by Rob for discretizing a column
# data_series is continuous valued column to be discretized
# bins is the number of "levels" for discretization (default 2 = binary)
def discretize_column(data_series, bins=2):
    sorted_series = data_series.order(ascending=True, inplace=False)
    return_series = pandas.Series([0]*data_series.shape[0], dtype=int)
    for i in range(1, bins):
        split = sorted_series.iloc[sorted_series.shape[0]*i/bins]
        above_split = pandas.Series(data_series > split, dtype=int)
        return_series += np.array(above_split)
    return return_series


# A function for to be implemented
def basic_feature_selection(data_frame):
    feature_list = ["Level", "probDiff"]
    return feature_list, data_frame


# Function for testing feature selection methods
def testing_feature_selection(data_frame):
    full_list = list(data_frame.columns.values)
    test_feature = pandas.Series([0]*data_frame.shape[0],dtype=int)
    test_feature[data_frame["reward"] < 0] = 1
    data_frame["TEST_FEATURE"] = test_feature
    test_feature2 = discretize_column(data_frame["Interaction"],bins=3)
    data_frame["TEST_FEATURE2"] = test_feature2
    feature_list = ['Level', 'probDiff', "TEST_FEATURE", "TEST_FEATURE2"]
    return feature_list, data_frame


# Function for doing feature selection by PCA
def PCA_feature_selection(data_frame, vectors_used=8):
    feature_list = []
    return feature_list, None


if __name__ == "__main__":

    original_data = pandas.read_csv('MDP_Original_data2.csv')

    selected_features, expanded_data = testing_feature_selection(original_data)

    ECR_value = induce_policy_MDP2(expanded_data, selected_features)