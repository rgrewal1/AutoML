import sklearn.metrics
from autosklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import autosklearn.regression
import pickle

target_column = 'zone_4'
#Load trainig data and split in features and target columns
data = pd.read_csv(f'Dataset/Zones/{target_column}train.csv')
data = data.drop(['datetime'], axis=1)
features = data.drop('target', axis=1)
target = data['target']
#runtime in seconds
runtimes = [600,7200,14400]

if __name__ == "__main__":
    #run with different times
    for time in runtimes:
        #5-fold cross validation, 200GB max Memory, scoring metric= r2, 
        #n_jobs=1 due to unusual behavior. Better performance and less memory limit exceeds with n_jobs=1 
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=time,resampling_strategy= 'holdout', resampling_strategy_arguments={'train_size':0.8},
             n_jobs=1, memory_limit=2000000, metric = mean_absolute_error)
        
        #run auto.sklearn
        automl.fit(features, target, dataset_name= target_column)
        #quick stats
        print(automl.sprint_statistics())
        x = automl.show_models()
        results = automl # the regressor itself
        #save models in target directory 
        pickle.dump(results, open(f'Frameworks/Auto-sklearn/{target_column}/{target_column}_{time}','wb'))