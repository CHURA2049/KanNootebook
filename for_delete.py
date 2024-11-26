import pandas as pd
import matplotlib.pyplot as plt
# Способ обьединения столбцов даты и номера часа который предложил чатик и он работает, но и старый тоже работает

# Загрузка данных
dataFrame = pd.read_csv('file_for_input/all_hour/SBER_tradestats_test_hour.csv')
dataFrame = dataFrame[:7]

df = pd.DataFrame(dataFrame)

# Преобразование столбцов tradedate и tradetime_hour в формат datetime
df['tradedate'] = pd.to_datetime(df['tradedate'])
df['datetime'] = df['tradedate'] + pd.to_timedelta(df['tradetime_hour'], unit='h')

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(df['datetime'], df['pr_close_hour'], marker='o', linestyle='-', color='b', label='pr_close_hour')
plt.xlabel('Time')
plt.ylabel('pr_close_hour')
plt.title('pr_close_hour over Time')
plt.legend()
plt.grid(True)
plt.show()
