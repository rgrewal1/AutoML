from datetime import datetime
from matplotlib import pyplot as plt
from Utils.preprocess_data import DataHandler
from Utils.validate_model import ModelValidation

if __name__ == "__main__":
    target_column = 'zone_1'

    dh = DataHandler()
    dh.load_data()
    dh.shift_data(target_column=target_column, target_name='target', drop_others=True)
    df_load_temp, df_load, df_temp = dh.create_calendar_features(continent='usa', country='UnitedStates',
                                                                 holiday=True, bridgeday=False, weekday=False)

    df_load[target_column].plot()
    plt.show()

    df_load_temp.to_csv(f"Dataset/{target_column}.csv")  # todo: funktion überprüfen

    mv = ModelValidation(data=df_load_temp)
    data_train, data_val = mv.split_train_val(val_start=datetime(year=2008, month=7, day=1, hour=0),
                                              val_end=datetime(year=2008, month=7, day=7, hour=23))


    data_train.to_csv(f"Dataset/{target_column}training.csv")
    data_val.to_csv(f"Dataset/{target_column}testing.csv")
    # split your data into X and y before training the model
    # if the target column ins present in X, the model will just learn the identity to y
    # train the model with the automl framework
    # select best-performing model -> refit auf allen train daten
    # model = autoMLFramework.fit(X, y)

    error = mv.calculate_error(model=model, target_column=target_column, target_name='target', metric='MAE')
