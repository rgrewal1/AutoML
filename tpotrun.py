
from tpot import TPOTRegressor
import pandas as pd
import pickle

#set zone and the metric
target_zone = 'zone_4'
data = pd.read_csv(f'Dataset/Zones/{target_zone}train.csv')
#drop datetime as TPOT doesn't support this format
data = data.drop(['datetime'], axis=1)
features = data.drop('target', axis=1)
target = data['target']
#run for 10min, 2h and 4h
runtimes = [10, 120, 240]

if __name__ == "__main__":
    for time in runtimes:
        tpot = TPOTRegressor( n_jobs=48, max_time_mins=time, verbosity=2, random_state=42)
        tpot.fit(features, target)
        #tpot.export(f'Frameworks/TPOT/{target_zone}/{target_zone}_{time*60}.py')
        #save the best pipeline refitted on the whole data
        model = tpot.fitted_pipeline_
        pickle.dump(model, open(f'Frameworks/TPOT/{target_zone}/{target_zone}_{time*60}','wb'))