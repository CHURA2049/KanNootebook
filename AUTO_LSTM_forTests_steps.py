import datetime
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
import math
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(99)
tf.compat.v1.enable_eager_execution() #используете Eager execution (режим немедленного выполнения
from sklearn.metrics import *
import csv


# neurons_number= [60,30,20,10,5,1] - 200,150, 100

# steps = [ 50, 30, 20]  - 100,


# ПАРАМЕТРЫ
secid = 'TASB'
neurons_number = 1
window = 17
num_param= 13
drop_out= 0.0
steps = 5
name = secid+ '_lstm_' + 'n'+str(neurons_number) + 'win'+str(window)+ 'd'+ str(drop_out).replace('.','_')+'s'+ str(steps) +'np' + str(num_param) + '.csv'

num_lerning = 2











with open(name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['rmse_lstm', 'mae_lstm', 'mape_lstm', 'r2_lstm', 'msle_lstm'])


# Здесь НАЧНЁМ подготваливать свои данные
dataFrame = pd.read_csv('file_for_input/all_hour/TASB_tradestats_test_hour.csv ')
dataFrame['tradedate'] = dataFrame['tradedate'] + '-' + dataFrame['tradetime_hour'].astype(str)
dataFrame = dataFrame.drop('tradetime_hour', axis=1)
dataFrame = dataFrame.rename(columns={'tradedate': 'Date'})
dataFrame = dataFrame.rename(columns={'pr_close_hour': 'Close'})
dataFrame = dataFrame.rename(columns={'pr_open_hour': 'Open'})
print(dataFrame)
print(dataFrame.columns)
print(dataFrame.head(n=2))
print("KONEC -----------------------------------")

# Здесь ЗАКОНЧИМ подготваливать свои данные

#Data preprocessing
imputer = SimpleImputer(missing_values=np.nan) # Handling missing values
dataFrame.drop(columns=['Date'], inplace=True)
dataFrame = pd.DataFrame(imputer.fit_transform(dataFrame), columns=dataFrame.columns)
dataFrame = dataFrame.reset_index(drop=True)
# Applying feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(dataFrame.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(dataFrame.columns))
target_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled[['Open', 'Close']] = target_scaler.fit_transform(dataFrame[['Open', 'Close']].to_numpy())
df_scaled = df_scaled.astype(float)
# df_scaled = dataFrame.astype(float)


# Здесь начнём добавлять тестовую выборку
# вычисляем количество строк, которое соответствует 10% от общего количества строк
n = int(len(dataFrame) * 0.10)
# выбираем последние n строк
df_test = dataFrame.iloc[-n:]
dataFrame= dataFrame.iloc[:-n] # Убираем последние n% строк из основного датасета
# Здесь нЗакончим создавать тестовую выборку






# Single step dataset preparation
def singleStepSampler(df, window):
	xRes = []
	yRes = []
	for i in range(0, len(df) - window):
		res = []
		for j in range(0, window):
			r = []
			for col in df.columns:
				r.append(df[col][i + j])
			res.append(r)
		xRes.append(res)
		yRes.append(df[['Open', 'Close']].iloc[i + window].values)
	return np.array(xRes), np.array(yRes)

# Dataset splitting
SPLIT = 0.85
(xVal, yVal) = singleStepSampler(df_scaled, window) # было (xVal, yVal) = singleStepSampler(df_scaled, 20)
print("Кол-во данных после сплитинга ",len(xVal),len(yVal))


X_final_test = xVal[int(SPLIT * len(xVal)):]
y_final_test = yVal[int(SPLIT * len(yVal)):]
xVal = xVal[:int(SPLIT * len(xVal))]
yVal = yVal[:int(SPLIT * len(yVal))]
print("длина X_final_test,y_final_test ",len(X_final_test),len(y_final_test))
print("Кол-во данных после X_final_test ",len(xVal),len(yVal))


X_train = xVal[:int(SPLIT * len(xVal))]
y_train = yVal[:int(SPLIT * len(yVal))]
X_test = xVal[int(SPLIT * len(xVal)):]
y_test = yVal[int(SPLIT * len(yVal)):]
print("(обычное обучение)Кол-во данных  X_train, y_train ",len(X_train),len(y_train))
print("(обычное обучение)Кол-во данных  X_test, y_test ",len(X_test),len(y_test))



multivariate_lstm = keras.Sequential()
multivariate_lstm.add(keras.layers.LSTM(neurons_number, input_shape=(X_train.shape[1], X_train.shape[2])))
multivariate_lstm.add(keras.layers.Dropout(drop_out))
multivariate_lstm.add(keras.layers.Dense(2, activation='linear'))
multivariate_lstm.compile(loss='mae', metrics=['mse'], optimizer='adam') # Было: metrics=['mae']
multivariate_lstm.summary()
for i in 1:
	history = multivariate_lstm.fit(X_train, y_train, epochs=steps)#,validation_split=0.4

	# Forecast Plot with Dates on X-axis
	predicted_values = multivariate_lstm.predict(X_test)

	d = {
		'Predicted_Open': predicted_values[:, 0],
		'Predicted_Close': predicted_values[:, 1],
		'Actual_Open': y_test[:, 0],
		'Actual_Close': y_test[:, 1],
	}

	d = pd.DataFrame(d)
	d.index = dataFrame.index[-len(y_test):] # Assigning the correct date index

	# fig, ax = plt.subplots(figsize=(10, 6))
	# highlight the forecast
	# highlight_start = int(len(d) * 0.9)
	# highlight_end = len(d) - 1 # Adjusted to stay within bounds
	# Plot the actual values
	# plt.plot(d[['Actual_Close']][:highlight_start], label=['Actual_Close'])
	#
	# # Plot predicted values with a dashed line
	# plt.plot(d[['Predicted_Close']], label=['Predicted_Close'], linestyle='--')
	#
	# # Highlight the forecasted portion with a different color
	# plt.axvspan(d.index[highlight_start], d.index[highlight_end], facecolor='lightgreen', alpha=0.5, label='Forecast')
	#
	# #  ГРАфик предсказаных значений
	# plt.title('LSTM')
	# plt.xlabel('Dates')
	# plt.ylabel('Values')
	# ax.legend()
	# plt.show()
	#
	#
	#
	# Построение графика зависимости потерь от эпохи
	plt.figure(figsize=(10, 6))
	plt.plot(history.history['loss'], label='Training Loss')
	plt.plot(history.history['mae'], label='Training MSE')
	plt.title('Model Training')
	plt.ylabel('Loss/MAE')
	plt.xlabel('Epoch')
	plt.legend()
	plt.show()






	# Model Evaluation  МЕТРИКИ МОДЕЛИ
	def eval(model):
		return {
			'MSE': sklearn.metrics.mean_squared_error(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy()),
			'MAE': sklearn.metrics.mean_absolute_error(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy()),
			'R2': sklearn.metrics.r2_score(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy())
		}

	result = dict()

	for item in ['Predicted_Close']:
		result[item] = eval(item)

	print(result)

	# комментарий
	# Начинаем работу с X_final_test Y_final_test

	y_pred_final_test = multivariate_lstm.predict(X_final_test) # Предсказываем значения


	# Оценка для ДИПЛОМА
	rmse_lstm = root_mean_squared_error(y_final_test, y_pred_final_test)
	mae_lstm = mean_absolute_error(y_final_test, y_pred_final_test)
	mape_lstm = mean_absolute_percentage_error(y_final_test, y_pred_final_test)
	r2_lstm = r2_score(y_final_test, y_pred_final_test)
	msle_lstm = mean_squared_log_error(y_final_test, y_pred_final_test)


	# оздайте DataFrame для хранения фактических и предсказанных значений:

	d_final_test = pd.DataFrame({
		'Actual_Open': y_final_test[:, 0],
		'Actual_Close': y_final_test[:, 1],
		'Predicted_Open': y_pred_final_test[:, 0],
		'Predicted_Close': y_pred_final_test[:, 1]
	})

	# Рисуем график фактических и предсказанных значений:
	plt.figure(figsize=(10, 6))

	plt.plot(d_final_test['Actual_Close'], label='Actual Close')
	plt.plot(d_final_test['Predicted_Close'], label='Predicted Close')

	plt.title('LSTM Model - Actual vs. Predicted')
	plt.xlabel('Time')
	plt.ylabel('Close Price')
	plt.legend()
	plt.show()

	with open('file_for_output/hour_metrics/LSTM/'+name, 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow([rmse_lstm, mae_lstm, mape_lstm, r2_lstm, msle_lstm])


print(" Название сохранённого файла: "+ name)