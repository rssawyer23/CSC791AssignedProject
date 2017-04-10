# File for converting the output policy from Shitian's code into conditional probabilities on one state to save space

FILENAME = "PolicyOutput.txt"
STATES = 8


def read_policy(filename, states):
    list_dict = [dict() for _ in range(states)]

    policy_file = open(filename, "r")
    for line in policy_file:
        space_split = line.split(" ")
        state_split = space_split[0].split(":")
        for s_index in range(len(state_split)):
            if state_split[s_index] not in list_dict[s_index].keys():
                list_dict[s_index][state_split[s_index]] = [0,0]
            if space_split[2] == "PS,":
                list_dict[s_index][state_split[s_index]][0] += 1
            list_dict[s_index][state_split[s_index]][1] += 1
    return list_dict


def print_reduced_policy(list_dict, states):
    for state_index in range(states):
        for key in list_dict[state_index].keys():
            percent = list_dict[state_index][key][0] / float(list_dict[state_index][key][1])
            print "STATE NUMBER:%d STATE VARIABLE:%s PERCENT PS:%.4f" % (state_index, key, percent)

list_dict = read_policy(FILENAME, STATES)
print_reduced_policy(list_dict, STATES)
