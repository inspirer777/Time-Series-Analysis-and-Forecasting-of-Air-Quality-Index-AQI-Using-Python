import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from time import time
#######
import pandas as pd          
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series
import warnings                   # To ignore the warnings
import os
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math



#############
humidity = pd.read_csv('C:/Users/BEHINLAPTOP/Desktop/11.csv', index_col='date', parse_dates=['date'])
humidity.head()

humidity = humidity.iloc[1:]
humidity = humidity.fillna(method='ffill')

humidity["AQI"].plot() 
plt.title('AQI')
plt.show()

humidity["AQI"].plot(legend=True)
shifted = humidity["AQI"].shift(10).plot(legend=True)
shifted.legend(['AQI','AQI_lag'])
plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
##  acf
plot_acf(humidity["AQI"],lags=25,title="AQI")
plt.show()

## pacf
plot_pacf(humidity["AQI"],lags=25)
plt.show()
###############


#####
humidity.diff().plot(figsize=(20,6))
plt.show()


###
my_order = (0,1,0)
my_seasonal_order = (1, 0, 1, 12)
# define model
train_data = humidity["AQI"]*2

model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)

start = time()
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)

print(model_fit.summary())




