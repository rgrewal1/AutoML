import pandas as pd
import numpy as np
from datetime import timedelta
import workalendar.africa
import workalendar.america
import workalendar.asia
import workalendar.europe
import workalendar.oceania
import workalendar.usa
import datetime


class DataHandler:

    def __init__(self,
                 directory: str = None,
                 filenames_load: list = None,
                 filenames_temp: list = None):

        self._directory = 'Dataset/' if directory is None else directory
        self._filenames_load = ['Load_history.csv',
                                'Load_solution.csv'] if filenames_load is None else filenames_load
        self._filenames_temp = ['temperature_history.csv',
                                'temperature_solution.csv'] if filenames_temp is None else filenames_temp

        self._df_load = None
        self._df_temp = None
        self._df_load_temp = None
        self._calendar = None

    @property
    def df_load(self):
        """Returns the load dataframe"""
        return self._df_load

    @property
    def df_temp(self):
        """Returns the temperature dataframe"""
        return self._df_temp

    def load_data(self):
        """Loads the load and temperature dataframe"""

        load_dfs, temp_dfs = [], []
        for (file_load, file_temp) in zip(self._filenames_load, self._filenames_temp):
            load_dfs.append(self._process_data(f"{self._directory}{file_load}"))
            temp_dfs.append(self._process_data(f"{self._directory}{file_temp}"))

        self._df_load = self._concat_history_solution(load_dfs)
        self._df_temp = self._concat_history_solution(temp_dfs)

        # merge load and temperature into one dataframe
        self._df_load_temp = self._df_temp.copy()
        zones = [zone for zone in self._df_load if zone.startswith('zone')]
        self._df_load_temp[zones] = self._df_load[zones]

        return self._df_load_temp, self._df_load, self._df_temp

    def shift_data(self, target_column: str, target_name: str = "target", drop_others: bool = False):
        """Creates the target column by shifting the data by one time step"""

        # create shift
        self._df_load[target_name] = self.df_load[target_column].shift(-1)
        self._df_load_temp[target_name] = self._df_load_temp[target_column].shift(-1)

        # drop nan-values in the last row
        self._df_load.drop(index=self._df_load.index[-1], inplace=True)
        self._df_load_temp.drop(index=self._df_load_temp.index[-1], inplace=True)

        if drop_others:
            zones = [col for col in self._df_load if col.startswith('zone') and col != target_column]
            self._df_load.drop(columns=zones, inplace=True)
            self._df_load_temp.drop(columns=zones, inplace=True)

        return self._df_load_temp, self._df_load, self._df_temp

    def create_calendar_features(self, continent: str = 'europe', country: str = 'UnitedKingdom',
                                 holiday: bool = True, bridgeday: bool = True, weekday: bool = True):
        """Create boolean features to differentiate between working days and weekend/holiday/bridge days"""

        # initialize calendar
        self._calendar = self._init_calendar(continent=continent, country=country)

        # create boolean features
        if holiday and bridgeday:
            self._df_load['workday'] = self._df_load.index.to_series().apply(
                lambda x: (self._is_holiday_or_bridgeday(x)) * 1)
            self._df_load_temp['workday'] = self._df_load_temp.index.to_series().apply(
                lambda x: (self._is_holiday_or_bridgeday(x)) * 1)
        elif holiday:
            self._df_load['workday'] = self._df_load.index.to_series().apply(
                lambda x: (self._calendar.is_working_day(x)) * 1)
            self._df_load_temp['workday'] = self._df_load_temp.index.to_series().apply(
                lambda x: (self._calendar.is_working_day(x)) * 1)
        elif bridgeday:
            raise SyntaxError("Bridging days required considering of holidays")
        else:
            self._df_load['workday'] = self._df_load.index.to_series().apply(
                lambda x: (x.weekday() <= 5) * 1)
            self._df_load_temp['workday'] = self._df_load_temp.index.to_series().apply(
                lambda x: (x.weekday() <= 5) * 1)

        if weekday:
            self._df_load['weekday'] = self._df_load.index.to_series().apply(lambda x: x.weekday())
            self._df_load_temp['weekday'] = self._df_load.index.to_series().apply(lambda x: x.weekday())

        # todo: add sin/cos encoding option
        return self._df_load_temp, self._df_load, self._df_temp

    # todo: add function to create lag features

    def _is_holiday_or_bridgeday(self, time):

        if self._calendar.is_working_day(time):
            if time.weekday() == 0:  # Monday, bridge day if free on Tuesday
                if self._calendar.is_holiday(time + datetime.timedelta(days=1)):
                    return False
            elif time.weekday() == 4:  # Friday, bridge day if free on Thursday
                if self._calendar.is_holiday(time - datetime.timedelta(days=1)):
                    return False
            return True
        return False

    @staticmethod
    def _init_calendar(continent: str, country: str):
        """Check if continent and country are correct and return calendar object."""
        if hasattr(workalendar, continent.lower()):
            module = getattr(workalendar, continent.lower())
            if hasattr(module, country):
                return getattr(module, country)()
            else:
                raise SyntaxError(f"The country {country} does not fit to the continent {continent}")
        else:
            raise SyntaxError(f"The continent {continent} does not exist.")

    @staticmethod
    def _concat_history_solution(dfs):
        """concatenates the dataframes of history and solution"""

        # concatenate history and solution dataframes and sort
        df = pd.concat(dfs)
        df.sort_index(inplace=True)

        # check if gaps are present in the datetime index
        deltas = df.index.to_series().diff()[1:]
        gaps = deltas[deltas > timedelta(hours=1)]
        if len(gaps) > 1:  # there is one gap of 19 hours on 2008-06-30
            raise ValueError("Identified gaps in the datetime index")

        # fill 19h gap
        df = df.asfreq('1h', fill_value=np.nan)

        # fix total load
        if 'zone_21' in df.columns:
            df.drop(columns=['zone_21'], inplace=True)  # drop zone 21
            zones = [col for col in df if col.startswith('zone')]
            df['zone_total'] = df[zones].sum(axis=1)  # calculate sum over all zones

        return df

    @staticmethod
    def _process_data(filename):
        """Re-organizes the dataframes"""

        df = pd.read_csv(filename, thousands=',')

        if 'zone_id' in df.columns:
            ids = pd.unique(df['zone_id'])
            id_col = 'zone_id'
        elif 'station_id' in df.columns:
            ids = pd.unique(df['station_id'])
            id_col = 'station_id'
        else:
            raise ValueError("No _id found in dataframe!")

        dfs = []
        for id_ in ids:

            if id_col == 'zone_id':
                id_str = f"zone_{id_}"
            elif id_col == 'station_id':
                id_str = f"station_{id_}"
            data = {
                'year': [],
                'month': [],
                'day': [],
                'hour': [],
                id_str: []
            }
            for index, row in df.loc[df[id_col] == id_].iterrows():

                for key in data.keys():

                    if 'h1' in df.columns:

                        if key == 'hour':  # add zero padding
                            data['hour'] = [*data['hour'], *[str(int(hour)).zfill(2) for hour in range(24)]]
                        elif key in ['day', 'month']:  # add zero padding
                            data[key] = [*data[key], *[str(int(row[key])).zfill(2)] * 24]
                        elif key == 'year':
                            data[key] = [*data[key], *[str(int(row[key]))] * 24]
                        elif key == f"zone_{id_}" or key == f"station_{id_}":
                            data[id_str] = [*data[id_str], *row[[f"h{hour + 1}" for hour in range(24)]]]

                    else:  # temp_solution

                        if key == 'hour':  # add zero padding
                            data[key].append(str(int(row[key] - 1)).zfill(2))
                        elif key in ['day', 'month']:  # add zero padding
                            data[key].append(str(int(row[key])).zfill(2))
                        elif key == 'year':
                            data[key].append(str(int(row[key])))
                        elif key == f"zone_{id_}" or key == f"station_{id_}":
                            data[id_str].append(row['T0_p1'])

            dfs.append(pd.DataFrame(data))

        # check if year, month, day, and hour are equal
        date = ['year', 'month', 'day', 'hour']
        if all(x[date].equals(dfs[0][date]) for x in dfs):
            df = dfs[0]  # overwrite original dataframe

            for i, id_ in enumerate(ids[1:], 1):
                if id_col == 'zone_id':
                    id_str = f"zone_{id_}"
                elif id_col == 'station_id':
                    id_str = f"station_{id_}"
                df[id_str] = dfs[i][id_str]

            df['datetime'] = pd.to_datetime(
                df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour'],
                infer_datetime_format=True)
            df = df.set_index(df['datetime'])  # assign datetime as index
            df.drop(columns=['datetime'], inplace=True)

            # convert str date features to numeric features to be processable by learning algorithm
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)
            df['hour'] = df['hour'].astype(int)

            # drop rows with gaps
            df.dropna(inplace=True)

        else:
            raise ValueError("Dates between zones or stations does not match")

        return df
