import pandas as pd
from MDP_function2 import discretize_column

base_features = ['student','currProb','course','session','priorTutorAction','reward']
features = ['CurrPro_avgProbTimeWE', 'NextStepClickCountWE', 'cumul_TotalWETime', 'ruleScoreCD', 'ruleScoreADD', 'ruleScoreASSOC', 'difficultProblemCountWE', 'easyProblemCountWE']
bins = [5, 6, 5, 5, 2, 2, 2, 4]

data = pd.read_csv("MDP_Original_data2.csv")
new_data = pd.DataFrame()
for f,b in zip(features,bins):
    new_data["New-%s"%f] = discretize_column(data[f],bins=b)
new_data.columns = features
final_df = pd.concat([data[base_features], new_data], axis=1,ignore_index=True)
final_df.columns = base_features + features
final_df.to_csv("Training_data.csv", index=False)
