import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool, cv
from typing import Union
from datetime import datetime


class AtlasModel(object):

    def __init__(self, rsm=0.6, learning_rate=0.013, depth=6):
        self.params = {
            'loss_function': 'MAE',
            'iterations': 3000,
            'depth': depth,
            'rsm': rsm,
            'random_seed': 0,
            'learning_rate': learning_rate
        }
        self.cb_temp1_model = CatBoostRegressor(n_estimators=3000,
                                                learning_rate=self.params['learning_rate'],
                                                rsm=self.params['rsm'],
                                                depth=self.params['depth'],
                                                loss_function=self.params['loss_function'],
                                                thread_count=-1,
                                                random_seed=0,
                                                logging_level='Silent')
        self.cb_temp2_model = self.cb_temp1_model.copy()
        self.cb_temp3_model = self.cb_temp1_model.copy()
        return

    def fit(self, data_list: list, n_estimators: Union[int, None] = None):

        train_data, target_temp1, target_temp2, target_temp3 = self.__prepare_train_data(data_list)

        # Calculate number of trees
        if n_estimators is None:
            n_estimators_1 = self.__calculate_n_estimators(train_data, target_temp1)
            n_estimators_2 = self.__calculate_n_estimators(train_data, target_temp2)
            n_estimators_3 = self.__calculate_n_estimators(train_data, target_temp3)
        else:
            n_estimators_1 = n_estimators
            n_estimators_2 = round(n_estimators / 5)
            n_estimators_3 = n_estimators

        # Set params
        self.cb_temp1_model.set_params(n_estimators=n_estimators_1)
        self.cb_temp2_model.set_params(n_estimators=n_estimators_2)
        self.cb_temp3_model.set_params(n_estimators=n_estimators_3)

        # Fit models
        self.cb_temp1_model.fit(train_data, target_temp1)
        self.cb_temp2_model.fit(train_data, target_temp2)
        self.cb_temp3_model.fit(train_data, target_temp3)

        return self

    def predict(self, X):
        prediction_data = self.__prepare_data(X)
        return np.vstack((self.cb_temp1_model.predict(prediction_data) + X['ai1 mean'],
                          self.cb_temp2_model.predict(prediction_data) + X['ai2 mean'],
                          self.cb_temp3_model.predict(prediction_data) + X['ai3 mean']
                          )).T

    def __prepare_train_data(self, data_list: list):
        train_data = pd.DataFrame()
        target_temp1 = np.empty(shape=[0, ])
        target_temp2 = np.empty(shape=[0, ])
        target_temp3 = np.empty(shape=[0, ])
        for dataset in data_list:
            # Prepare synthetic features
            train_data = pd.concat([train_data, self.__prepare_data(dataset)[50:]], axis=0, ignore_index=True)
            # Target for model = the next temperature value minus the current value
            target_temp1 = np.hstack(
                (target_temp1,
                 (dataset['ai1 mean'].shift(-1, fill_value=dataset['ai1 mean'].iloc[-1]) -
                  dataset['ai1 mean']).values[50:]))
            target_temp2 = np.hstack(
                (target_temp2,
                 (dataset['ai2 mean'].shift(-1, fill_value=dataset['ai2 mean'].iloc[-1]) -
                  dataset['ai2 mean']).values[50:]))
            target_temp3 = np.hstack(
                (target_temp3,
                 (dataset['ai3 mean'].shift(-1, fill_value=dataset['ai3 mean'].iloc[-1]) -
                  dataset['ai3 mean']).values[50:]))
        return train_data, target_temp1, target_temp2, target_temp3

    def __prepare_data(self, X):
        data = X.copy()
        # Add volatility features
        data = data.join(self.__generate_volatility(data))
        # Add lag features
        data = data.join(self.__generate_lags(data))
        # Add one-hot encoding of "Mode"
        data = data.join(self.__generate_mode_ohe(data))
        # Add cycle uptime
        data = data.join(self.__generate_uptime(data))

        # Differential features
        data = data.join(self.__generate_diffs(data))
        # EMA-smoothed temperature change and volatility rates on each of the 3 sensors
        data = data.join(self.__generate_ema(data['Temp1_diff1'], np.array([2.5, 9, 30]),
                                             ['Temp1_ema_short', 'Temp1_ema_avg', 'Temp1_ema_long']))
        data = data.join(self.__generate_ema(data['Ai1_volat'], np.array([18]),
                                             ['Volat1_ema']))
        data = data.join(self.__generate_ema(data['Temp2_diff1'], np.array([2]),
                                             ['Temp2_ema_short']))
        data = data.join(self.__generate_ema(data['Ai2_volat'], np.array([6]),
                                             ['Volat2_ema']))
        data = data.join(self.__generate_ema(data['Temp3_diff1'], np.array([1.7, 5, 15]),
                                             ['Temp3_ema_short', 'Temp3_ema_avg', 'Temp3_ema_long']))
        data = data.join(self.__generate_ema(data['Ai3_volat'], np.array([10]),
                                             ['Volat3_ema']))
        # Cooling rate for sensors (normalized to the temperature difference in the workshop and in tank No.1)
        data = data.join(self.__generate_cooling_down(data))

        # Drop unnecessary columns
        data.drop(columns=['Ai1 min', 'ai1 max', 'Ai2 min', 'ai2 max', 'Ai3 min', 'ai3 max',
                           'Mode', 'Ai1_volat', 'Ai2_volat', 'Ai3_volat'], axis=1, inplace=True)
        return data

    def __calculate_n_estimators(self, train_data, target):
        cv_data = cv(
            params=self.params,
            pool=Pool(train_data, label=target),
            fold_count=3,
            shuffle=False,
            partition_random_seed=0,
            plot=False,
            stratified=False,
            verbose=False
        )
        return np.argmin(cv_data['test-MAE-mean'])

    # Generate volatility features
    def __generate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:

        result = pd.DataFrame()

        result['Ai1_volat'] = data['ai1 max'] - data['Ai1 min']
        result['Ai2_volat'] = data['ai2 max'] - data['Ai2 min']
        result['Ai3_volat'] = data['ai3 max'] - data['Ai3 min']

        return result

    # Generate lag features
    def __generate_lags(self, data: pd.DataFrame) -> pd.DataFrame:

        result = pd.DataFrame()

        result['Temp1_shift1'] = data['ai1 mean'].shift(1, fill_value=data['ai1 mean'].iloc[0])
        result['Temp1_shift2'] = data['ai1 mean'].shift(2, fill_value=data['ai1 mean'].iloc[0])
        result['Temp3_shift1'] = data['ai3 mean'].shift(1, fill_value=data['ai3 mean'].iloc[0])

        return result

    # Generate mode one hot encoding features
    def __generate_mode_ohe(self, data: pd.DataFrame) -> pd.DataFrame:

        result = pd.DataFrame()

        result['Mode1'] = (data.Mode == 1) * 1
        result['Mode2'] = (data.Mode == 2) * 1
        result['Mode3'] = (data.Mode == 3) * 1
        result['Mode4'] = (data.Mode == 4) * 1

        return result

    # Generate uptime features
    def __generate_uptime(self, data: pd.DataFrame) -> pd.DataFrame:

        result = pd.DataFrame()

        result['Cycle_uptime'] = np.nan

        cycle_time = 0
        current_mode1 = 0
        current_mode2 = 0
        current_mode3 = 0
        current_mode4 = 0

        for index, row in data.iterrows():
            cycle_time += 1
            if (current_mode1 == 0 and row['Mode1'] == 1) or \
                    (current_mode2 == 0 and row['Mode2'] == 1) or \
                    (current_mode3 == 0 and row['Mode3'] == 1) or \
                    (current_mode4 == 0 and row['Mode4'] == 1):
                cycle_time = 1
            current_mode1 = row['Mode1']
            current_mode2 = row['Mode2']
            current_mode3 = row['Mode3']
            current_mode4 = row['Mode4']
            result.loc[index, 'Cycle_uptime'] = cycle_time

        # Double weight of "Cycle_uptime" feature
        result['Cycle_uptime1'] = result['Cycle_uptime']

        return result

    def __generate_ema(self, data: pd.Series, periods: np.array, columns: list) -> pd.DataFrame:
        alphas = 2 / (periods + 1)

        # Memory allocation for resulting values
        result_ema = np.ndarray((data.shape[0], len(alphas)), dtype=float)

        # Fill buffer vectors
        ema_buffer = np.ones((len(alphas),), dtype=float) * data.iloc[0]
        result_ema[0] = ema_buffer

        # Calculation of EMA, DMA, TMA, QMA
        for i in range(1, data.shape[0]):
            ema_buffer = ema_buffer * (1 - alphas) + alphas * data.iloc[i]
            result_ema[i] = ema_buffer

        # Convert result into pandas DaraFrame
        return pd.DataFrame(result_ema, index=data.index, columns=columns)

    # Generate differential features
    def __generate_diffs(self, data: pd.DataFrame) -> pd.DataFrame:

        diff_data = pd.DataFrame()

        diff_data['Temp1_diff1'] = data['ai1 mean'] - data['ai1 mean'].shift(1, fill_value=data['ai1 mean'].iloc[0])
        diff_data['Temp1_diff3'] = data['ai1 mean'] - data['ai1 mean'].shift(3, fill_value=data['ai1 mean'].iloc[0])
        diff_data['Temp2_diff1'] = data['ai2 mean'] - data['ai2 mean'].shift(1, fill_value=data['ai2 mean'].iloc[0])
        diff_data['Temp3_diff1'] = data['ai3 mean'] - data['ai3 mean'].shift(1, fill_value=data['ai3 mean'].iloc[0])
        diff_data['Temp3_diff3'] = data['ai3 mean'] - data['ai3 mean'].shift(3, fill_value=data['ai3 mean'].iloc[0])

        return diff_data

    # Generate cooling down features
    def __generate_cooling_down(self, data: pd.DataFrame) -> pd.DataFrame:

        result = pd.DataFrame()

        result['Temp1_cooling_down'] = (data['Temp1_diff3'] / (data['Temp1_ema_long'] - 23))
        result['Temp2_cooling_down'] = (data['Temp2_diff1'] / (data['Temp2_ema_short'] - 23))

        return result


def make_augmentations(data: pd.DataFrame) -> list:
    data_aug1 = data.copy()
    data_aug1['Mode'] = data_aug1['Mode'].shift(-1, fill_value=data['Mode'].iloc[-1])
    data_aug1['ai1 mean'] = data_aug1['ai1 mean'].shift(1, fill_value=data['ai1 mean'].iloc[0])

    data_aug2 = data.copy()
    data_aug2['Mode'] = data_aug2['Mode'].shift(1, fill_value=data['Mode'].iloc[0])
    data_aug2['ai1 mean'] = data_aug2['ai1 mean'].shift(-1, fill_value=data['ai1 mean'].iloc[-1])

    data_aug3 = data.copy()
    data_aug3['ai1 mean'] = data_aug3['Ai1 min'] * 0.75 + data_aug3['ai1 mean'] * 0.25
    data_aug3['ai2 mean'] = data_aug3['Ai2 min']
    data_aug3['ai3 mean'] = data_aug3['ai3 max'] * 0.75 + data_aug3['ai3 mean'] * 0.25

    data_aug4 = data.copy()
    data_aug4['ai1 mean'] = data_aug4['ai1 max'] * 0.75 + data_aug4['ai1 mean'] * 0.25
    data_aug4['ai2 mean'] = data_aug4['ai2 max']
    data_aug4['ai3 mean'] = data_aug4['Ai3 min'] * 0.75 + data_aug4['ai3 mean'] * 0.25

    return [data, data_aug1, data_aug2, data_aug3, data_aug4, data]


# Read data from excel
row_data = pd.read_excel('DataFl72_reup.xlsx', index_col=0)
row_data = row_data.reset_index(level=[0]).drop('#', axis=1)

# Separate data on three folds and drop zero rows
data_fold1 = row_data.iloc[:1308].copy()
data_fold1 = data_fold1.drop(data_fold1[(data_fold1.min(axis=1) < 1)].index)
data_fold2 = row_data.iloc[1308:4700].copy()
data_fold2 = data_fold2.drop(data_fold2[(data_fold2.min(axis=1) < 1)].index)
data_fold3 = row_data.iloc[4680:].copy()
data_fold3 = data_fold3.drop(data_fold3[(data_fold3.min(axis=1) < 1)].index)

# Calculate target values for validation
fold1_target1 = (data_fold1['ai1 mean'].shift(-1, fill_value=data_fold1['ai1 mean'].iloc[-1])).values[50:]
fold1_target2 = (data_fold1['ai2 mean'].shift(-1, fill_value=data_fold1['ai2 mean'].iloc[-1])).values[50:]
fold1_target3 = (data_fold1['ai3 mean'].shift(-1, fill_value=data_fold1['ai3 mean'].iloc[-1])).values[50:]

fold2_target1 = (data_fold2['ai1 mean'].shift(-1, fill_value=data_fold2['ai1 mean'].iloc[-1])).values[50:]
fold2_target2 = (data_fold2['ai2 mean'].shift(-1, fill_value=data_fold2['ai2 mean'].iloc[-1])).values[50:]
fold2_target3 = (data_fold2['ai3 mean'].shift(-1, fill_value=data_fold2['ai3 mean'].iloc[-1])).values[50:]

fold3_target1 = (data_fold3['ai1 mean'].shift(-1, fill_value=data_fold3['ai1 mean'].iloc[-1])).values[50:]
fold3_target2 = (data_fold3['ai2 mean'].shift(-1, fill_value=data_fold3['ai2 mean'].iloc[-1])).values[50:]
fold3_target3 = (data_fold3['ai3 mean'].shift(-1, fill_value=data_fold3['ai3 mean'].iloc[-1])).values[50:]


'''# CROSS VALIDATION #
for rsm in [0.4, 0.5, 0.6, 0.75, 0.92]:
    for learning_rate in [0.01, 0.013, 0.016]:
        print(f'depth={rsm}, learning_rate={learning_rate}')
        AvgMAE = 0.0
        # FOLD №1 #
        # Get data augmentations
        augmented_data = make_augmentations(data_fold2) + make_augmentations(data_fold3)

        # Fit-predict
        model = AtlasModel(learning_rate=learning_rate, rsm=rsm)
        model.fit(augmented_data, n_estimators=2000)
        predicted_values = model.predict(data_fold1)[50:]

        print('Data fold #1')
        print(f'Наибольшая ошибка прогноза Temp1={np.max(np.abs(predicted_values[:, 0] - fold1_target1)):.3f}°C, '
              f'MAE={mean_absolute_error(predicted_values[:, 0], fold1_target1):.3f}°C')
        print(f'Наибольшая ошибка прогноза Temp2={np.max(np.abs(predicted_values[:, 1] - fold1_target2)):.3f}°C, '
              f'MAE={mean_absolute_error(predicted_values[:, 1], fold1_target2):.3f}°C')
        print(f'Наибольшая ошибка прогноза Temp3={np.max(np.abs(predicted_values[:, 2] - fold1_target3)):.3f}°C, '
              f'MAE={mean_absolute_error(predicted_values[:, 2], fold1_target3):.3f}°C')
        print()
        AvgMAE += mean_absolute_error(predicted_values[:, 0], fold1_target1)*0.5 + \
                  mean_absolute_error(predicted_values[:, 1], fold1_target2)*0.5 + \
                  mean_absolute_error(predicted_values[:, 2], fold1_target3)*0.5

        # FOLD №2 #
        # Get data augmentations
        augmented_data = make_augmentations(data_fold1) + make_augmentations(data_fold3)

        # Fit-predict
        model = AtlasModel(learning_rate=learning_rate, rsm=rsm)
        model.fit(augmented_data, n_estimators=2000)
        predicted_values = model.predict(data_fold2)[50:]

        print('Data fold #2')
        print(f'Наибольшая ошибка прогноза Temp1={np.max(np.abs(predicted_values[:, 0] - fold2_target1)):.3f}°C, '
              f'RMSE={mean_absolute_error(predicted_values[:, 0], fold2_target1):.3f}°C')
        print(f'Наибольшая ошибка прогноза Temp2={np.max(np.abs(predicted_values[:, 1] - fold2_target2)):.3f}°C, '
              f'RMSE={mean_absolute_error(predicted_values[:, 1], fold2_target2):.3f}°C')
        print(f'Наибольшая ошибка прогноза Temp3={np.max(np.abs(predicted_values[:, 2] - fold2_target3)):.3f}°C, '
              f'RMSE={mean_absolute_error(predicted_values[:, 2], fold2_target3):.3f}°C')
        print()
        AvgMAE += mean_absolute_error(predicted_values[:, 0], fold2_target1) + \
                  mean_absolute_error(predicted_values[:, 1], fold2_target2) + \
                  mean_absolute_error(predicted_values[:, 2], fold2_target3)

        # FOLD №3 #
        # Get data augmentations
        augmented_data = make_augmentations(data_fold1) + make_augmentations(data_fold2)

        # Fit-predict
        model = AtlasModel(learning_rate=learning_rate, rsm=rsm)
        model.fit(augmented_data, n_estimators=2000)
        start_fit_time = datetime.now()
        predicted_values = model.predict(data_fold3)[50:]

        print('Data fold #3')
        print(f'Наибольшая ошибка прогноза Temp1={np.max(np.abs(predicted_values[:, 0] - fold3_target1)):.3f}°C, '
              f'MAE={mean_absolute_error(predicted_values[:, 0], fold3_target1):.3f}°C')
        print(f'Наибольшая ошибка прогноза Temp2={np.max(np.abs(predicted_values[:, 1] - fold3_target2)):.3f}°C, '
              f'MAE={mean_absolute_error(predicted_values[:, 1], fold3_target2):.3f}°C')
        print(f'Наибольшая ошибка прогноза Temp3={np.max(np.abs(predicted_values[:, 2] - fold3_target3)):.3f}°C, '
              f'MAE={mean_absolute_error(predicted_values[:, 2], fold3_target3):.3f}°C')
        print()
        AvgMAE += mean_absolute_error(predicted_values[:, 0], fold3_target1) + \
                  mean_absolute_error(predicted_values[:, 1], fold3_target2) + \
                  mean_absolute_error(predicted_values[:, 2], fold3_target3)
        AvgMAE /= 7.5
        print(f'Average MAE={AvgMAE:.4f}')
        print(f'Время инференса одной строки {(datetime.now() - start_fit_time).total_seconds()/data_fold3.shape[0]:.4f}\n')'''


# DYNAMIC #

data_fold23 = row_data.iloc[1308:].copy()
data_fold23 = data_fold23.drop(data_fold23[(data_fold23.min(axis=1) < 1)].index)
fold23_target1 = (data_fold23['ai1 mean'].shift(-1, fill_value=data_fold23['ai1 mean'].iloc[-1])).values
fold23_target2 = (data_fold23['ai2 mean'].shift(-1, fill_value=data_fold23['ai2 mean'].iloc[-1])).values
fold23_target3 = (data_fold23['ai3 mean'].shift(-1, fill_value=data_fold23['ai3 mean'].iloc[-1])).values

model = AtlasModel()
augmented_data = make_augmentations(data_fold23)
model.fit(augmented_data, n_estimators=2000)
start_fit_time = datetime.now()
predicted_values = model.predict(data_fold23)

error1_values = predicted_values[:, 0] - fold23_target1
error2_values = predicted_values[:, 1] - fold23_target2
error3_values = predicted_values[:, 2] - fold23_target3

errors_target = pd.DataFrame(np.vstack((error1_values, error2_values, error3_values)).T,
                             columns=['error1_values', 'error2_values', 'error3_values'])

errors_target.to_csv('errors_target')
data_fold23.to_csv('errors_data')




