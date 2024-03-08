from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


target_column = 'zone_18'
#Load Zone training dataset
data = pd.read_csv(f'Dataset/Zones/{target_column}train.csv')
data = data.drop(['datetime'], axis=1)
train_data = TabularDataset(data)
#runtime in seconds
runtimes = [600,7200,14400]

if __name__ == "__main__":
    for time in runtimes:
        automl = TabularPredictor(label= 'target', problem_type='regression', eval_metric= 'mean_absolute_error')

        predictor = automl.fit(train_data, time_limit = time, refit_full='best',holdout_frac=0.2,  keep_only_best=True, save_space=True, ag_args_fit={'num_gpus': 0})

        pickle.dump(predictor, open(f'Frameworks/AutoGluon/{target_column}/{target_column}_{time}new','wb'))





