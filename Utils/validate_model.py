import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelValidation:

    def __init__(self, data: pd.DataFrame = None):
        self._data = data
        self._data_train = None
        self._data_val = None

    def split_train_val(self, val_start: datetime, val_end: datetime, data: pd.DataFrame = None):
        if data is not None:
            self._data = data

        self._data_train = self._data.drop(self._data.loc[val_start:val_end].index).copy()
        self._data_val = self._data.loc[val_start:val_end].copy()

        # drop nan-values, learning algorithms cannot handle nan's
        self._data_train.dropna(inplace=True)
        self._data_val.dropna(inplace=True)

        return self._data_train, self._data_val

    def calculate_error(self, model, target_column: str, target_name: str, metric: str,
                        data_train: pd.DataFrame = None, data_val: pd.DataFrame = None):

        if data_train is not None:
            self._data_train = data_train
        if data_val is not None:
            self._data_val = data_val

        # prediction loop
        date_index = self._data_train.index[-1]
        X = self._data.drop([target_column])[date_index]
        y_predict = []
        while date_index != self._data_val.index[-1]:  # todo: check if while loop exits correctly

            y_predict.append(model.predict(X))

            date_index += timedelta(hours=1)  # todo: check if this works
            X = self._data.drop([target_column])[date_index]
            X[target_column] = y_predict[-1]  # replace actual value by prediction

        # calculate error
        y_true = self._data_val[target_name]
        y_predict = y_predict[len(y_predict)-len(y_true):]  # remove the 19 values before the challenge week
        # todo: test with actual data, maybe the lengths need to be adapted

        error = None
        if metric == 'MSE':
            error = mean_squared_error(y_true, y_predict)
        elif metric == 'RMSE':
            error = mean_squared_error(y_true, y_predict, squared=False)
        elif metric == 'MAE':
            error = mean_absolute_error(y_true, y_predict)
        elif metric == 'r2':
            error = r2_score(y_true, y_predict)

        return error
