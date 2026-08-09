import fastf1 as f1 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

CACHE_DIR="cache"
YEAR=2023
EVENT='bahrain'
SESSION_TYPE='r'
FEATURES = [
    "StintLap",
    "LapNumber",
    "RollingAvg",
    "Compound",
    "Driver"
]

f1.Cache.enable_cache(CACHE_DIR)
session=f1.get_session(YEAR,EVENT,SESSION_TYPE)
session.load()
laps=session.laps
clean_laps=laps.copy()
clean_laps=clean_laps[~clean_laps['PitInTime'].notna()]
clean_laps=clean_laps[~clean_laps['PitOutTime'].notna()]
clean_laps=clean_laps[clean_laps["TrackStatus"]=='1']
clean_laps=clean_laps.dropna(subset=['LapTime'])
clean_laps = clean_laps.drop(columns =['FastF1Generated','IsAccurate','Deleted','DeletedReason'])
clean_laps['LapTimeSeconds']=clean_laps['LapTime'].dt.total_seconds()
clean_laps['LapNumber']=clean_laps['LapNumber'].astype(int)
clean_laps['StintLap']=clean_laps.groupby(['Driver','Stint']).cumcount() + 1
clean_laps['RollingAvg']=(clean_laps.groupby("Driver")['LapTimeSeconds'].rolling(window=4,min_periods=1).mean().reset_index(level=0,drop=True))
clean_laps["DeltaFromRolling"]=(clean_laps['LapTimeSeconds']-clean_laps['RollingAvg'])
print(clean_laps['Compound'].value_counts())
driver1=clean_laps[clean_laps["Driver"]=='VER']
driver2=clean_laps[clean_laps["Driver"]=='PER']
driver1 = driver1.dropna(subset=['LapTime'])
driver2 = driver2.dropna(subset=['LapTime'])

x=clean_laps[FEATURES]
y=clean_laps['LapTimeSeconds']
x=pd.get_dummies(x,columns=['Compound','Driver'],drop_first=True)
x.info()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print('mae',mae)
print('rmse',rmse)
plt.scatter(y_test,y_pred,s=10)
plt.xlabel('actual lap time ')
plt.ylabel('pred lap time ')
plt.title('Baseline Model: Actual vs Predicted')
plt.show()
coef_df = pd.DataFrame({
    'feature': x.columns,
    'coefficient': model.coef_
}).sort_values(by='coefficient')
rf=RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)
mae_rf=mean_absolute_error(y_test,y_pred_rf)
rmse_rf=np.sqrt(mean_squared_error(y_test,y_pred_rf))
print('mae rf',mae_rf)
print('rmse rf ',rmse_rf)
res_lin=y_test-y_pred
res_rf=y_test-y_pred_rf
res_df=x_test.copy()
res_df['res_lin']=res_lin
res_df['res_rf']=res_rf
res_df[['res_lin','res_rf']].describe()
plt.scatter(res_df['StintLap'],res_df['res_lin'],s=10,color='Red')
plt.scatter(res_df['StintLap'],res_df['res_rf'],s=10,color='Blue')
plt.axhline(0)
plt.xlabel('StintLap')
plt.ylabel("residual")
plt.title('residual lin(red) vs rf(blue)')
plt.show()
feature_importance=pd.DataFrame({
    'features':x.columns,
    'Importance':rf.feature_importances_
}).sort_values(by='Importance',ascending=False)
print(feature_importance)
driver_cols=[c for c in x.columns if c.startswith('Driver_')]
def get_driver(row):
    for col in driver_cols:
        if row[col]==1:
            return col.replace('Driver_','')
    return 'Baseline'
res_df['Driver']=res_df.apply(get_driver,axis=1)
driver_res=(res_df.groupby('Driver')
            .agg(
                lin_mean=('res_lin','mean'),
                lin_std=('res_lin','std'),
                rf_mean=('res_rf','mean'),
                rf_std=('res_rf','std'),
                laps=('res_lin','count')
            ).sort_values('lin_std'))
print(driver_res)
driver_res[['lin_std','rf_std']].plot(kind='bar')
plt.ylabel('res std')
plt.title('driver consistency  lin vs rf ')
plt.show()
clean_laps=clean_laps.sort_values(by=['Driver','LapNumber'])
split_lap=int(clean_laps['LapNumber'].max()*0.8)
train_df=clean_laps[clean_laps['LapNumber']<=split_lap]
test_df=clean_laps[clean_laps['LapNumber']>split_lap]
features=[
    'StintLap',
    'LapNumber','RollingAvg','Compound','Driver'
]
x_train=train_df[features]
y_train=train_df['LapTimeSeconds']
x_test=test_df[features]
y_test=test_df['LapTimeSeconds']
x_train=pd.get_dummies(x_train,columns=["Compound","Driver"],drop_first=True)
x_test=pd.get_dummies(x_test,columns=['Compound','Driver'],drop_first=True)
x_test=x_test.reindex(columns=x_train.columns,fill_value=0)
model.fit(x_train,y_train)
y_pred_lin=model.predict(x_test)
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)
mae_lin=mean_absolute_error(y_test,y_pred_lin)
rmse_lin=np.sqrt(mean_squared_error(y_test,y_pred_lin))
print('mae lin ',mae_lin)
print('rmse lin',rmse_lin)
mae_rf=mean_absolute_error(y_test,y_pred_rf)
rmse_rf=np.sqrt(mean_squared_error(y_test,y_pred_rf))
print('mea rf',mae_rf)
print('rmse_rf',rmse_rf)
plt.figure(figsize=(10,5))
plt.scatter(test_df["LapNumber"],y_test-y_pred_lin,s=10,color='Red')
plt.scatter(test_df['LapNumber'],y_test-y_pred_rf,s=10,color='Blue')
plt.axhline(0)
plt.xlabel('Lap')
plt.ylabel('Residual')
plt.show()
late_stint_thres=15
late_stint=res_df[res_df['StintLap']>=late_stint_thres]
late_risk_lin=(
    late_stint.groupby('Driver').agg(
        mean_res=('res_lin','mean'),
        std_res=('res_lin','std'),
        spike_rate=('res_lin',lambda x:(x.abs()>0.8).mean()),
        laps=('res_lin','count')
).sort_values('std_res'))
print(late_stint.head())
late_risk_lin[['std_res','spike_rate']].plot(kind='bar')
plt.ylabel('Risk metric linear')
late_risk_rf=(late_stint.groupby('Driver').agg(  
    mean_res=('res_rf','mean'),
    std_res=('res_rf','std'),
    spike_rate=('res_rf',lambda x:(x.abs()>0.8).mean()),
    laps=('res_rf','count')
).sort_values('std_res')
)
late_risk_rf[['std_res','spike_rate']].plot(kind='bar')
plt.ylabel('rf rm')
plt.show()