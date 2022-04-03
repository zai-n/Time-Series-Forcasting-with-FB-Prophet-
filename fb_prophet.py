import warnings; 
warnings.simplefilter('ignore')

"""# 0. Install and Import Dependencies"""

!pip install pystan
!pip install fbprophet

import pandas as pd
from fbprophet import Prophet

"""# 1. Read in Data and Process Dates"""

df = pd.read_csv('dataset.csv')

df['Year'] = df['Time Date'].apply(lambda x: str(x)[-4:])
df['Month'] = df['Time Date'].apply(lambda x: str(x)[-6:-4])
df['Day'] = df['Time Date'].apply(lambda x: str(x)[:-6])
df['ds'] = pd.DatetimeIndex(df['Year']+'-'+df['Month']+'-'+df['Day'])

df = df.loc[(df['Product']==2667437) & (df['Store']=='QLD_CW_ST0203')]
df.drop(['Time Date', 'Product', 'Store', 'Year', 'Month', 'Day'], axis=1, inplace=True)
df.columns = ['y', 'ds']

df.head()

"""# 2. Train Model"""

mod = Prophet(interval_width=0.95, daily_seasonality=True)
model = mod.fit(df)

"""# 3. Forecast Away"""

future = mod.make_future_dataframe(periods=100,freq='D')
forecast = mod.predict(future)
forecast.head()

plot1 = mod.plot(forecast)

plt2 = mod.plot_components(forecast)

import pickle
with open('saved_model.pkl', "wb") as f:
    pickle.dump(mod, f)
