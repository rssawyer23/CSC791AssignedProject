import collections
import numpy as np
import pandas
import mdptoolbox, mdptoolbox.example
import argparse
import sklearn.decomposition
from scipy import stats


def generate_MDP_input2(original_data, features):

    students_variables = ['student', 'priorTutorAction', 'reward']

    # generate distinct state based on feature
    original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    # original_data['state'] = original_data[features].apply(tuple, axis=1)
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
    # distinct_states didn't contain terminal state
    student_list = list(data['student'].unique())
    distinct_states = list()
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        # don't consider last row
        temp_states = list(student_data['state'])[0:-1]
        distinct_states = distinct_states + temp_states
    distinct_states = list(set(distinct_states))

    Ns = len(distinct_states)

    # we include terminal state
    start_states = np.zeros(Ns + 1)
    A = np.zeros((Nx, Ns+1, Ns+1))
    expectR = np.zeros((Nx, Ns+1, Ns+1))

    # update table values episode by episode
    # each episode is a student data set
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        # count the number of transition among states without terminal state
        for i in range(1, (len(row_list)-1)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

        # count the number of transition from state to terminal
        state1 = distinct_states.index(student_data.loc[row_list[-2], 'state'])
        act = student_data.loc[row_list[-1], 'priorTutorAction']
        A[act, state1, Ns] += 1
        expectR[act, state1, Ns] += float(student_data.loc[row_list[-1], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        A[act, Ns, Ns] = 1
        # generate expected reward
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        # some states only have either PS or WE transition to other state
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act, l, l] = 1
            
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
        print(distinct_states[s] + " -> " + distinct_acts[vi.policy[s]] + ", " + str(vi.V[s]))

def induce_policy_MDP2(original_data, selected_features):

    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    # output policy
    output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    print('ECR value: ' + str(ECR_value))
    return ECR_value


# Function implemented by Rob for discretizing a column
# data_series is continuous valued column to be discretized
# bins is the number of "levels" for discretization (default 2 = binary)
def discretize_column(data_series, bins=2):
    sorted_series = data_series.sort_values(ascending=True, inplace=False)
    return_series = pandas.Series([0]*data_series.shape[0], dtype=int)
    for i in range(1, bins):
        split = sorted_series.iloc[sorted_series.shape[0]*i/bins]
        above_split = pandas.Series(data_series > split, dtype=int)
        return_series += np.array(above_split)
    return return_series


# A function for to be implemented
def basic_feature_selection(data_frame, bins=2):
    feature_list = ["CurrPro_avgProbTimeWE"]
    data_frame["CurrPro_avgProbTimeWE"] = discretize_column(data_frame["CurrPro_avgProbTimeWE"], bins)
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


# Function for performing feature selection via Principal Component Analysis (dimensionality reduction)
# vectors_used is number of component vectors (resulting features) to use
# bins is the number of levels to discretize each resulting feature vector into
# uniform controls whether each feature vector has the same amount of levels or whether level amount should decrease (less important vectors later)
def PCA_feature_selection(data_frame, vectors_used=8, bins=2, uniform=True):
    pca = sklearn.decomposition.PCA(n_components=vectors_used)
    transformed_data = pca.fit_transform(data_frame.iloc[:, 5:])
    feature_list = ["New-Feature-%d" % i for i in range(vectors_used)]
    extra_data = pandas.DataFrame(transformed_data, columns=feature_list)
    result_data = pandas.concat([data_frame, extra_data], axis=1)
    if uniform:
        for f in feature_list:
            result_data[f] = discretize_column(result_data[f], bins)
    else:
        for f,i in zip(feature_list,range(len(feature_list))):
            new_bins = max(bins - i, 2)
            result_data[f] = discretize_column(result_data[f], new_bins)
    return feature_list, result_data


# Replicating their "initialization" of selecting best feature
# Runs incredibly slowly since solves MDP for each feature
def identify_best_feature(data_frame, initial_features, bins=2, uniform=False):
    feature_list = list(data_frame.columns.values)
    end_dict = dict()
    for f in feature_list:
        end_dict[f] = []
    best_ECR = 0
    best_feature = ""
    best_bins = ""
    offset = 6
    for b in range(bins):
        for feature, i in zip(list(data_frame.iloc[:,offset:].columns.values),range(data_frame.shape[1]-offset)):
            if i%50 == 0:
                print "Bins:%d Feature:%d" % (bins, i)
            data_frame["single_feature"] = discretize_column(data_frame[feature], bins=b)
            for f, bin_i in zip(initial_features, range(len(initial_features))):
                if not uniform:
                    data_frame[f] = discretize_column(data_frame[f],bins=bins)
                else:
                    data_frame[f] = discretize_column(data_frame[f],bins=bins+(len(initial_features)-bin_i))
            ECR_value = induce_policy_MDP2(data_frame, initial_features+["single_feature"])
            end_dict[feature].append(ECR_value)
            # try:
            #     ECR_value = induce_policy_MDP2(data_frame,["single_feature"])
            # except OverflowError:
            #     print "Overflow %s" % feature
            #     print "Bins:%d Feature:%d" % (bins, i)
            if ECR_value > best_ECR:
                best_ECR = ECR_value
                best_feature = feature
                best_bins = b
    return best_feature, best_ECR, best_bins, end_dict

if __name__ == "__main__":

    original_data = pandas.read_csv('MDP_Original_data2.csv')

    # selected_features, expanded_data = PCA_feature_selection(original_data, vectors_used=8, bins=4, uniform=False)
    #
    # ECR_value = induce_policy_MDP2(expanded_data, selected_features)

    for i in range(2,11):
        selected_features, expanded_data = basic_feature_selection(original_data, bins=i)
        ECR_value = induce_policy_MDP2(expanded_data, selected_features)

    # #Below code takes forever to run
    # for i in range(2, 9):
    #     print "Discretization Levels:%d" % i
    #     initial_features = []
    #     feature, ecr, bins, dictionary = identify_best_feature(original_data, initial_features, bins=i)
    #     print "Best Feature:%s for level:%d at ECR:%.5f with bins:%d" % (feature, i, ecr, bins)
    #     pandas.DataFrame(dictionary).to_csv("MDP_BEST_FEATURE_OUTPUT.csv", index=False)
