import collections
import numpy as np
import pandas
import mdptoolbox, mdptoolbox.example
import argparse
import sklearn.decomposition
from scipy import stats
from sklearn.cluster import KMeans


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
# Transition probabilities calculated here
# A is table that would need to be sampled
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

    Counts = A.copy()
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

    return [start_states, A, expectR, distinct_acts, distinct_states, Counts]


def calcuate_ECR(start_states, expectV):
    ECR_value = start_states.dot(np.array(expectV))
    return ECR_value


def output_policy(distinct_acts, distinct_states, vi):
    Ns = len(distinct_states)
    # print('Policy: ')
    # print('state -> action, value-function')
    # for s in range(Ns):
    #     print(distinct_states[s] + " -> " + distinct_acts[vi.policy[s]] + ", " + str(vi.V[s]))

def induce_policy_MDP2(original_data, selected_features):

    [start_states, A, expectR, distinct_acts, distinct_states, Counts] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP

    vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    # output policy
    output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    #print('ECR value: ' + str(ECR_value))
    return ECR_value


# Function for sampling a transition probability matrix from counts of transitions
# Uses a Dirichlet distribution
def sample_transition_probs(Count_matrix, distinct_acts):
    sampled_matrix = np.zeros(Count_matrix.shape)
    for action in range(len(distinct_acts)):
        for row, row_index in zip(Count_matrix[action], range(Count_matrix[action].shape[0])):
            try:
                new_row = list(np.random.dirichlet(row))
            except ZeroDivisionError:
                new_row = [0] * len(row)
            sampled_matrix[action, row_index] = np.array(new_row)

        for l in np.where(np.sum(sampled_matrix[action], axis=1) == 0)[0]:
            sampled_matrix[action, l, l] = 1
    return sampled_matrix


def calculate_confidence_interval(original_data, selected_features, samples=1000):
    [start_states, A, expectR, distinct_acts, distinct_states, Counts] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP
# Can input a sampling A statement here, wrap over this, including ECR value

    # A = Sample from Counts as Dirichlet, make this first line of loop, keep list of ECR values
    ECR_values = []
    for i in range(samples):
        sampled_A = sample_transition_probs(Counts, distinct_acts)
        vi = mdptoolbox.mdp.ValueIteration(sampled_A, expectR, 0.9)
        vi.run()

        # evaluate policy using ECR
        ECR_value = calcuate_ECR(start_states, vi.V)
        ECR_values.append(ECR_value)
    return np.mean(ECR_values), np.std(ECR_values)

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


def discretize_column_clustering(data_series,clusters=2):
    kmeans = KMeans(n_clusters=clusters).fit(np.array(data_series).reshape(-1,1))
    return_series = kmeans.predict(np.array(data_series).reshape(-1,1))
    return return_series


# A function for to be implemented
def basic_feature_selection(data_frame, bins=2):
    feature_list = ["New_CurrPro_avgProbTimeWE"]
    data_frame["New_CurrPro_avgProbTimeWE"] = discretize_column_clustering(data_frame["CurrPro_avgProbTimeWE"], bins)
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
def PCA_feature_selection(data_frame, vectors_used=8, bins=2, uniform=True, discrete="percentiles"):
    pca = sklearn.decomposition.PCA(n_components=vectors_used)
    transformed_data = pca.fit_transform(data_frame.iloc[:, 5:])
    feature_list = ["New-Feature-%d" % i for i in range(vectors_used)]
    extra_data = pandas.DataFrame(transformed_data, columns=feature_list)
    result_data = pandas.concat([data_frame, extra_data], axis=1)
    if uniform:
        for f in feature_list:
            if discrete == "clusters":
                result_data[f] = discretize_column_clustering(result_data[f], bins)
            else:
                result_data[f] = discretize_column(result_data[f], bins)
    else:
        for f,i in zip(feature_list,range(len(feature_list))):
            new_bins = max(bins - i, 2)
            if discrete == "clusters":
                result_data[f] = discretize_column_clustering(result_data[f], new_bins)
            else:
                result_data[f] = discretize_column(result_data[f], new_bins)
    return feature_list, result_data


# Replicating their "initialization" of selecting best feature
# Runs incredibly slowly since solves MDP for each feature
def identify_best_feature(data_frame, initial_features, prev_bins, bins=2, uniform=False):
    if len(initial_features) > 1:
        feature_list = []
        for e in list(data_frame.iloc[:,5:].columns.values):
            if e not in initial_features:
                feature_list.append(e)
        #feature_list = [e for e in list(data_frame.iloc[:, 5:].columns.values) not in initial_features]
    else:
        feature_list = list(data_frame.iloc[:, 5:].columns.values)
    end_dict = dict()
    for f in feature_list:
        end_dict[f] = []
    best_ECR = 0
    best_feature = ""
    best_bins = ""
    if not uniform:
        try:
            max_bins = max(prev_bins) - 2
        except ValueError:
            max_bins = bins
    else:
        max_bins = bins
    for b in range(2,2+max_bins):
        for feature in feature_list:
            data_frame["single_feature"] = discretize_column(data_frame[feature], bins=b)

            for f, bin_i in zip(initial_features, range(len(initial_features))):
                data_frame[f] = discretize_column(data_frame[f], bins=prev_bins[bin_i])

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
                #print "ECR:%.5f, Feature:%s, bins:%d" % (best_ECR, best_feature, best_bins)

    return best_feature, best_ECR, best_bins, end_dict


def final_feature_selection(data, features, bins):
    new_features = []
    for f,b in zip(features,bins):
        data["New-%s"%f] = discretize_column(data[f],bins=b)
        new_features.append("New-%s"%f)
    return new_features, data


if __name__ == "__main__":

    original_data = pandas.read_csv('extractedfeatures2.csv')

    # selected_features, expanded_data = PCA_feature_selection(original_data, vectors_used=2, bins=4, uniform=False, discrete="clusters")
    #
    # ECR_value = induce_policy_MDP2(expanded_data, selected_features)
    # print "ECR:%.5f" % ECR_value

    # Below section for testing a specific feature across many bins
    # for i in range(2, 20):
    #     selected_features, expanded_data = basic_feature_selection(original_data, bins=i)
    #     ECR_value = induce_policy_MDP2(expanded_data, selected_features)
    #     print "Clusters:%d ECR:%.5f" % (i, ECR_value)

    # #Below section takes forever to run
    # for i in range(2, 11):
    #     print "Discretization Levels:%d" % i
    #     initial_features = []
    #     feature, ecr, bins, dictionary = identify_best_feature(original_data, initial_features, bins=i, uniform=False)
    #     print "Best Feature:%s for level:%d at ECR:%.5f with bins:%d" % (feature, i, ecr, bins)
    #     pandas.DataFrame(dictionary).to_csv("MDP_BEST_FEATURE_OUTPUT.csv", index=False)

    #Below section should be final code to run for gaming the system method of feature selection
    # initial_features = []
    # previous_bins = []
    # set_max_bins = True
    # max_bins = 5
    # output_file = open("MDP_Final_Output_%d.csv" % set_max_bins, mode='a')
    # output_file.write("Feature,Bins,ECR\n")
    # variance_output_file = open('MDP_Variance_Output.csv',mode='a')
    # variance_output_file.write("Mean,Std,Lower95,Upper95,Samples\n")
    # for i in range(1, 9):
    #     print "-------------------------------ITERATION %d--------------------" % i
    #     feature, ecr, bins, dictionary = identify_best_feature(original_data, initial_features, previous_bins, bins=max_bins, uniform=set_max_bins)
    #     print "Best Previous %d Features: %s" % (i-1, initial_features)
    #     print "Best New %d Feature:%s at ECR:%.5f with bins:%d" % (i, feature, ecr, bins)
    #     output_file.write("%s,%s,%.5f\n" % (feature, bins, ecr))
    #     initial_features.append(feature)
    #     previous_bins.append(bins)
    #     try:
    #         pandas.DataFrame(dictionary).to_csv("MDP_BEST_FEATURE_OUTPUT_%d_%d.csv" % (set_max_bins, i), index=False)
    #     except ValueError:
    #         print "Value Error"
    #
    # output_file.close()
    # # #Uses results from greedy search method to output final ECR
    # print initial_features
    # print previous_bins
    #
    # exp_data, new_feats = final_feature_selection(original_data, initial_features, previous_bins)
    # ECR_mean, ECR_std = calculate_confidence_interval(exp_data, new_feats, samples=10000)
    # variance_output_file.write("%.5f,%.5f,%.5f,%.5f,%d\n" % (ECR_mean, ECR_std, ECR_mean-1.96*ECR_std, ECR_mean+1.96*ECR_std,10000))
    # ECR of each one goes from 442, 430, 254, 161
    # potential_features = [['CurrPro_avgProbTimeWE', 'NextStepClickCountWE', 'cumul_TotalWETime', 'ruleScoreCD', 'ruleScoreADD', 'ruleScoreASSOC', 'difficultProblemCountWE', 'easyProblemCountWE'],
    #                      ['CurrPro_avgProbTimeWE', 'NextStepClickCountWE', 'cumul_TotalWETime', 'ruleScoreCD','ruleScoreADD','ruleScoreASSOC'],
    #                      ['CurrPro_avgProbTimeWE', 'NextStepClickCountWE', 'cumul_TotalWETime', 'ruleScoreCD','ruleScoreADD'],
    #                      ['CurrPro_avgProbTimeWE', 'NextStepClickCountWE', 'cumul_TotalWETime', 'ruleScoreCD']]
    # potential_bins = [[5, 6, 5, 5, 2, 2, 2, 4],
    #                   [5] * 4 + [2] * 2,
    #                   [5] * 4 + [2],
    #                   [5] * 4]
    potential_features = [['f7', 'f6', 'f1', 'f8', 'SolvedPSInLevel', 'cumul_TotalWETime', 'easyProblemCountSolved', 'cumul_OptionalCount']]
    potential_bins = [[6, 4, 3, 3, 3, 5, 2, 3]]
    variance_output_file = open('MDP_Variance_Output.csv',mode='a')
    variance_output_file.write("Mean,Std,Lower95,Upper95,Samples\n")
    for initial_features, previous_bins, index in zip(potential_features, potential_bins, range(len(potential_features))):
        for samples_used in [10000]:
            selected_features, expanded_data = final_feature_selection(original_data, initial_features, previous_bins)
            # ECR_Final = induce_policy_MDP2(expanded_data, selected_features)
            # print ECR_Final

            ECR_mean, ECR_std = calculate_confidence_interval(expanded_data, selected_features, samples=samples_used)
            print ECR_mean, ECR_std
            variance_output_file.write("%.5f,%.5f,%.5f,%.5f,%d\n" % (ECR_mean, ECR_std, ECR_mean-1.96*ECR_std, ECR_mean+1.96*ECR_std,samples_used))
    variance_output_file.close()