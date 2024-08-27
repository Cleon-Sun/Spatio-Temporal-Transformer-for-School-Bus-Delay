import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

# train_test = 'train' or 'test', train 有三年的数据，test有一年的数据
# 其他参数保持默认即可
# 目标变量是 Avg_Delay_Time or Delay_Count， 这里主要用Avg_Delay_Time
# lag_Time = 0, lag_Count=0, lstm_lag=0, 这三个参数是用来做lag的，在spatial temporal transformer 中，不用管这三个参数
# 需要先调用 import_data 函数，会输出一个dataframe，然后再调用 get_data_train AND get_data_test 函数，会输出pytorch tensor, 形状是（总周数，总学校数，总特征数据）

def import_data_new(file_path,train_test = 'train',weather='Yes',traffic='Yes',census='No',week_dummy='No',season_dummy='Yes',smart_week_dummy='No',target = 'Avg_Delay_Time',lag_Time = 0,lag_Count=0,lstm_lag=0):


    if train_test == 'train':
        data_18_21 = 'Yes'
        data_22 = 'No'
    else:
        data_18_21 = 'No'
        data_22 = 'Yes'

    lstm_lag_list = []
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=['lat','Year','Week_Number'])
    #print(df.columns)
    df['Year_week'] = df['Year'].astype(str) + '-' + df['Week_Number'].astype(str)
    df['Year_week'] = pd.to_datetime(df['Year_week'] + '-1', format='%Y-%W-%w')
    
    df = df.sort_values(by=['lat','Year_week'])
    df.drop(['School_Code', 'School_Name', 'address','owners', 'renters',  'value','male','other'],axis=1, inplace=True)
    df.drop(['PRCP_avg', 'PRCP_max','SNOW_avg', 'SNOW_max','WT08_sum','AWND_max','WT05_sum','WT01_sum','TMAX_max','SNWD_avg'],axis=1,inplace=True)
        
    
    if data_18_21 == 'Yes':
            df = df[df['Year'].isin([2018,2019,2021])]

    if data_22 == 'Yes':
        if lstm_lag!=0:
            lstm_lag_list = [52 - i for i in range(lstm_lag)]
        
        df2021 = df[df['Year'].isin([2021])]
        df2021 = df2021[df2021['Week_Number'].isin(lstm_lag_list)]
        df2022 = df[df['Year'].isin([2022])]
        df = pd.concat([df2021,df2022],ignore_index=True)
        

    if weather == 'No':
        df.drop(['AWND_avg', 
        'SNWD_max', 'TMAX_avg', 
       'TMIN_avg', 'TMIN_max', 'WSF2_avg', 'WSF2_max', 'WSF5_avg', 'WSF5_max',
       'WT02_sum', 'WT03_sum', 'WT04_sum', 'WT06_sum',
       ],axis=1,inplace=True)

    #print(df.columns)


    if census == 'No':
        df.drop([ 'female',
       'white', 'black', 'native', 'asian', 'islander',  'multi',
       'hispanic', 'highschool', 'college','income','rent'],axis=1,inplace=True)

    if traffic == 'No':
        df.drop(['Average_AADT', 'Average_DHV','Run_Type'],axis=1,inplace=True)
    else:
        run_dummies = pd.get_dummies(df['Run_Type'], drop_first=True)
        df.drop('Run_Type',axis=1,inplace=True)
        df = pd.concat([df, run_dummies], axis=1)
        df.columns = df.columns.astype('str')
        df.drop('other',axis=1,inplace=True)

    if week_dummy != 'No':
        time_fixed_effects = pd.get_dummies(df['Week_Number'], drop_first=True)
        df = pd.concat([df, time_fixed_effects], axis=1)
        df.columns = df.columns.astype('str')
        #df.drop(['25','12','45','2','47','19','51','13','22','23','48','24','5','6','10','16','14','20'],inplace=True,axis=1)


    if smart_week_dummy != 'No':
        df['ifholiday'] = df['Week_Number'].apply(lambda x: 1 if (x == 8 or x == 16 or (x >= 26 and x <= 35) or x == 52) else 0)

    if season_dummy !='No':
        df['spring'] = df['Week_Number'].apply(lambda x: 1 if ((x <= 26) and (x!=8)) else 0)
        df['summer'] = df['Week_Number'].apply(lambda x: 1 if ((x<=35) and (x>=27) or (x==52) or (x==8)) else 0)
        df['fall'] = df['Week_Number'].apply(lambda x: 1 if (x >= 36) and (x<52) else 0)
        
    if target == 'both':
        target = 'Avg_Delay_Time'
    
    elif target != 'both':
        target_list = ['Avg_Delay_Time','Delay_Count']
        target_list.remove(target)
        df.drop(target_list[0],inplace=True,axis=1)
        
        
    if lag_Time !=0:
        for i in range(1, lag_Time+1):
            df[f'Avg_Delay_Time{i}'] = df.groupby('lat')['Avg_Delay_Time'].shift(i)
            
    if lag_Count !=0:
        for i in range(1, lag_Count+1):
            df[f'Delay_Count{i}'] = df.groupby('lat')['Delay_Count'].shift(i)
            
            
    df = df.dropna()
    
    if season_dummy != 'No':
        cols = [col for col in df.columns if col not in ['spring', 'summer', 'fall']]
        cols += ['spring', 'summer', 'fall']
        df = df[cols]

    
    return df
def get_data_train(df_copy, target_name):
    df = df_copy.copy()
    
    df = df.sort_values(by=['lat','Year_week'])
    df.drop(['Year','Week_Number','tracts','long'],axis=1,inplace=True)
    
    target_list = ['Avg_Delay_Time','Delay_Count']
    target_list.remove(target_name)

    if target_list[0] in df.columns:
        df.drop(target_list[0], inplace=True, axis=1)
    
    cols = [col for col in df.columns if col != target_name]
    df = df[[target_name] + cols]
    df.columns = df.columns.astype(str)
    print(df.columns)
    exclude_columns = [target_name, 'Year_week','lat']
    
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    scaler = StandardScaler()
    scaler.fit(df[columns_to_scale])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    columns_to_pivot = [col for col in df.columns if col not in ['Year_week', 'lat']]

    # 使用pivot透视第一列
    arrays = [df.pivot(index='Year_week', columns='lat', values=col).values for col in columns_to_pivot]

    # 将所有的二维数组沿着第三个维度堆叠起来得到三维数组
    arr = np.stack(arrays, axis=2)
  
    data = torch.tensor(arr, dtype=torch.float32)
    return  data, scaler
def get_data_test(df_copy, target_name,scaler):



    df = df_copy.copy()
    
    df = df.sort_values(by=['lat','Year_week'])
    df.drop(['Year','Week_Number','tracts','long'],axis=1,inplace=True)
    
    target_list = ['Avg_Delay_Time','Delay_Count']
    target_list.remove(target_name)

    if target_list[0] in df.columns:
        df.drop(target_list[0], inplace=True, axis=1)
    
    cols = [col for col in df.columns if col != target_name]
    df = df[[target_name] + cols]
    df.columns = df.columns.astype(str)
    print(df.columns)
    exclude_columns = [target_name, 'Year_week','lat']
    
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    scaler.fit(df[columns_to_scale])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    columns_to_pivot = [col for col in df.columns if col not in ['Year_week', 'lat']]

    # 使用pivot透视第一列
    arrays = [df.pivot(index='Year_week', columns='lat', values=col).values for col in columns_to_pivot]

    # 将所有的二维数组沿着第三个维度堆叠起来得到三维数组
    arr = np.stack(arrays, axis=2)
  
    data = torch.tensor(arr, dtype=torch.float32)
    return  data

def get_data_train_no_X(df_copy, target_name):
    df = df_copy.copy()
    
    df = df.sort_values(by=['lat','Year_week'])
    df.drop(['Year','Week_Number','tracts','long'],axis=1,inplace=True)
    
    target_list = ['Avg_Delay_Time','Delay_Count']
    target_list.remove(target_name)

    if target_list[0] in df.columns:
        df.drop(target_list[0], inplace=True, axis=1)
    
    columns_to_pivot = [col for col in df.columns if col not in ['Year_week', 'lat']]

    # 使用pivot透视第一列
    arrays = [df.pivot(index='Year_week', columns='lat', values=col).values for col in columns_to_pivot]

    # 将所有的二维数组沿着第三个维度堆叠起来得到三维数组
    arr = np.stack(arrays, axis=2)
  
    data = torch.tensor(arr, dtype=torch.float32)

    # 计算除了最后一个轴的第一列以外所有列的均值和标准差
    mean = data[:, :, 1:].mean(dim=[0, 1], keepdim=True)
    std = data[:, :, 1:].std(dim=[0, 1], keepdim=True)

    # 标准化除了最后一个轴的第一列以外的所有列
    data[:, :, 1:] = (data[:, :, 1:] - mean) / (std + 1e-5)

    return  data



    df = df_copy.copy()
    
    df = df.sort_values(by=['lat','Year_week'])
    df.drop(['Year','Week_Number','tracts','long'],axis=1,inplace=True)
    
    target_list = ['Avg_Delay_Time','Delay_Count']
    target_list.remove(target_name)

    if target_list[0] in df.columns:
        df.drop(target_list[0], inplace=True, axis=1)
    
    cols = [col for col in df.columns if col != target_name]
    df = df[[target_name] + cols]
    df.columns = df.columns.astype(str)
    print(df.columns)
    exclude_columns = [target_name, 'Year_week','lat']
    
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    scaler.fit(df[columns_to_scale])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    columns_to_pivot = [col for col in df.columns if col not in ['Year_week', 'lat']]

    # 使用pivot透视第一列
    arrays = [df.pivot(index='Year_week', columns='lat', values=col).values for col in columns_to_pivot]

    # 将所有的二维数组沿着第三个维度堆叠起来得到三维数组
    arr = np.stack(arrays, axis=2)
  
    data = torch.tensor(arr, dtype=torch.float32)
    return  data

def get_adj(path):
    A = np.load(path)
    A_wave = A + np.eye(A.shape[0])
    D_wave = np.eye(A.shape[0]) * (np.sum(A_wave, axis=0) ** (-0.5))
    adj = np.matmul(np.matmul(D_wave, A_wave), D_wave)
    adj = adj[:, :32]
    adj = torch.Tensor(adj)
    

    return adj

if __name__ == '__main__':
    #有X的：
    file_path = 'school_weekly_weather_traffic2(1).csv'
    #df = import_data_new(file_path,train_test='train')
    #data_train,scaler = get_data_train(df,target_name='Avg_Delay_Time')
    #print(data_train.shape)

    #df_test = import_data_new(file_path,train_test='test')
    #data_test= get_data_test(df_test,target_name='Avg_Delay_Time',scaler=scaler)
    #print(data_test.shape)


    #只有Y的
    df_train = import_data_new(file_path,train_test='train', weather='No',traffic='No',season_dummy='No')
    data_train_noX = get_data_train_no_X(df_train, target_name='Avg_Delay_Time')
    print(data_train_noX.shape)


    df_test = import_data_new(file_path,train_test='test', weather='No',traffic='No',season_dummy='No')
    data_test_noX = get_data_train_no_X(df_test, target_name='Avg_Delay_Time')
    print(data_test_noX.shape)



