def model():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.style
    matplotlib.style.use('bmh')
    import seaborn as sns
    from IPython.display import display
    from numpy.linalg import LinAlgError
#Time series and ML modules
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    #!pip install pmdarima
    import pmdarima as pm
    from statsmodels.tsa.arima.model import ARIMA
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
    import warnings
    warnings.filterwarnings("ignore") 
    df = pd.read_csv('dataset/cinemaTicket_Ref.csv')
    df.head(2)
    df.shape 
    #Check for Null Values
    df.isnull().sum()
    df.dropna(inplace=True)
    df.isnull().sum()
    #Check for Duplicates
    print(f"Dataset has {df.duplicated().sum()} duplicated data")
    df.drop_duplicates(keep='first')
    def duplicate(df, column):
        if len(df[df[column].duplicated()]) == 0:
            print(f'STATUS: There are no duplicate values in the column of "{column}"')
        else:
            print(f'STATUS: There are {len(df[df[column].duplicated()])} duplicate values in the column of "{column}"')
    duplicate(df,'film_code')
    duplicate(df,'cinema_code')
    df['date'] = pd.to_datetime(df["date"],errors='coerce')
    cols = ['film_code','cinema_code','date']
    df['id']=df[cols].apply(lambda row:'_'.join(row.values.astype(str)), axis=1)
    df.head(2)
    #Reordering the dataset and bringing the 'id' column in the front to delete the duplicates
    df = df[['id'] + [a for a in df.columns if a != 'id'] ]
    df.tail(2)
    df.drop_duplicates('id',inplace=True)
    duplicate(df,'id')
    #finding the unique values
    unique_films = df['film_code'].nunique()
    unique_cinemas = df['cinema_code'].nunique()
    print(f"{unique_films} unique films.")
    print(f"{unique_cinemas} unique cinemas ")
    #Visualizing the distribution of total_Sales
    t_sales = df.total_sales.values
    ax = sns.displot(t_sales)
    fig = matplotlib.pyplot.gcf()#getting the current graph
    fig.set_size_inches(15.5 ,4.5)
    plt.title("Distribution of Total Sales")
    plt.xlabel('Total Sales')
    plt.xscale('log')
    plt.show()
    df_uni = df.copy()
    #bins in the range of 0-100m upto 1300000000
    bins = np.arange(0,1300000000,100000000)
    df_uni['binned']=pd.cut(df['total_sales'],bins)
    #creating a value count df with percentage of total sales in each bin
    vc_df = df_uni.reset_index().groupby(['binned']).size().to_frame('count')
    vc_df['percentage (%)'] = vc_df['count'].div(sum(vc_df['count'])).mul(100)
    vc_df = vc_df.sort_values(by=['percentage (%)'], ascending=False)
    vc_df = vc_df.reset_index()
    vc_df['binned']=vc_df['binned'].astype('str')
    #visualizing the distributions across the bins 
    x = vc_df['binned'].values
    y = vc_df['count'].values
    plt.bar(x,y)
    plt.ylabel('Number of Sales')
    plt.xlabel('Binned Total Sales')
    plt.title('Distribution of Total Sales across Bins')
    plt.xticks(rotation=90)
    #displaying the number of sales on top of the bins 
    for i,j in zip(x,y):
        label = "{:.1%}".format(j)
        plt.annotate(int(j),(i,j),textcoords='offset points',xytext=(0,2),ha='center',fontsize=10)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12.5, 4.5)
    plt.show()
    vc_df
    #Around 98% of the dataset is in the 1st bin.
    Q1 = df['total_sales'].quantile(.25) #25th quantile
    print(f"The first quantile is {Q1}")
    Q3 = df['total_sales'].quantile(0.75) #75th quantile
    print(f"The third quantile is {Q3}")
    IQR = Q3-Q1
    print(f"The IQR range is {IQR}")
    S = 1.5 * IQR #Threshold
    UB = Q3 + S #Upperbound - Anything above this is considered as an outlier
    LB = Q1 - S #Lowerbound - Anything below this is considered as an outlier
    print(f"Valid range for Total_Sales : {LB} <= Total Sales <= {UB}")
    #Empirical Rule
    mean = df['total_sales'].mean()
    print(f'mean is {mean}')
    std = df['total_sales'].std()
    print(f'standard deviation is {std}')
    if std>mean:
        print(f"standard deviation is {std/mean} higher than mean")
    #function to carry out the emperical formula     
    def empirical_rule(n=1 ,rt = False):
        UB = mean + (n* std)
        LB = mean - (n * std)
        if rt:
            return UB #returning Upper boundary 
        print(f"valid range for {n} std : {LB} <= Total Sales <= {UB}")
    empirical_rule(1) #returns 1 * std range
    empirical_rule(2) # returns 2 * std range
    empirical_rule(3) # returns 3 * std range
    #Removing outliers outside the upper bound of 3 * std as they would be considered as outliers
    UB = empirical_rule(3,rt=True)
    df2=df[df['total_sales']<= UB]
    print(f"Filtered total sales values <= {UB}")
    mean = df2['total_sales'].mean() 
    print(f"mean = {mean}")
    std = df2['total_sales'].std() 
    print(f"std  = {std}")
    print(f"standard deviation is {std/mean} higher than mean")
    #Visualizing the total sales distribution without the outliers
    t_sales = df2.total_sales.values
    ax =sns.displot(t_sales)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15.5 ,4.5)
    plt.title("Distribution of Total Sales")
    plt.xlabel('Total Sales')
    plt.xscale('log')
    plt.show()
    df_uni = df2.copy()
    #bins in the range of 0-10m upto Upper bound
    bins = np.arange(0,104219494,10000000)
    df_uni['binned']=pd.cut(df2['total_sales'],bins)
    #creating a value count df with percentage of total sales in each bin
    vc_df = df_uni.reset_index().groupby(['binned']).size().to_frame('count')
    vc_df['percentage (%)'] = vc_df['count'].div(sum(vc_df['count'])).mul(100)
    vc_df = vc_df.sort_values(by=['percentage (%)'], ascending=False)
    vc_df = vc_df.reset_index()
    vc_df['binned']=vc_df['binned'].astype('str')
    #visualizing the distributions across the bins 
    x = vc_df['binned'].values
    y = vc_df['count'].values
    plt.bar(x,y)
    plt.ylabel('Number of Sales')
    plt.xlabel('Binned Total Sales')
    plt.title('Distribution of Total Sales across Bins')
    plt.xticks(rotation=90)
    #displaying the number of sales on top of the bins 
    for i,j in zip(x,y):
        label = "{:.1%}".format(j)
        plt.annotate(int(j),(i,j),textcoords='offset points',xytext=(0,2),ha='center',fontsize=10)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12.5, 4.5)
    plt.show()
    vc_df #98% of the values from the 0-100m range which were actually in the 0-10m range
    # Defining a fucntion to return an aggregrate by time period 
    def aggregate_period(df, period,column, w_mean = False): 
        freq = df.date.dt.to_period(period)
        if w_mean == False:
            s = df.groupby(freq).agg({'total_sales': ['sum']})
            s.columns = ['met']
        else:
            s = df.groupby(freq).agg({column: ['mean']})
            s.columns = [f'met']
        s.reset_index(level=0, inplace=True)
        s.date= s.date.astype(str)
        return s
    #Defining a function to return an aggregrate by days of the week
    def aggregate_period_weekday(df,column, w_mean= False): 
        freq = df['date'].dt.day_name()
        if w_mean == False:
            s = df.groupby(freq).agg({'total_sales': ['sum']})
            s.columns = ['met']
        else:
            s = df.groupby(freq).agg({column: ['mean']})
            s.columns = [f'met']
        s.reset_index(level=0, inplace=True)
        s.date= s.date.astype(str)
        d = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        s = s.set_index('date').reindex(d).reset_index()
        return s
    #Defining a function to visualize the distributions across different time periods 
    def viz(df, title,n):
        date = df['date'][-n:] #the latest dates
        count_period =df['met'][-n:] #the latest sales
        plt.plot(date, count_period, linestyle='solid', color='black', marker='o')
        plt.title(f'{title}')
        plt.xlabel('Time Frequency')
        plt.ylabel("Total Sales")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12.5, 3.5)
        #to show the total sales 
        for x,y in zip(date,count_period):
            label = "{:.2e}".format(y)
            plt.annotate((label), # this is the text
                        (x,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(2,10), # distance from text to points (x,y)
                        ha='center' , fontsize=12) # horizontal alignment can be left, right or center
        overall_mean = count_period.mean()
        plt.fill_between(date, count_period,overall_mean,where=(count_period>overall_mean),alpha=0.10, interpolate=True, color='DarkGreen')
        plt.fill_between(date, count_period,overall_mean,where=(count_period<overall_mean),alpha=0.10, interpolate=True, color='Red')
        plt.legend(['Sales', 'above average', 'below average'], prop={'size': 8})
        plt.xticks(rotation=90)
        return plt.show()
    q = aggregate_period(df2, 'q','na')
    m = aggregate_period(df2, 'm','na')
    w = aggregate_period(df2, 'w','na')
    W = aggregate_period_weekday(df2,'na')
    viz(q ,"Total Sales by Quarter",25)
    viz(m ,"Total Sales by Month",25)
    viz(w ,"Total Sales by Week",15)
    viz(W ,"Total Sales by Week Day",15) 
    #Aggregrate by Day 
    df = aggregate_period(df2, 'd','na')
    df.set_index('date',inplace=True)
    df.rename(columns={'met':'Total_Sales'},inplace=True)
    df.sample(5)
    #Stationary Check - ADF TEST
    #Defining a adf calculator to check the stationary state of the dataset 
    def adf_calc(df, target):
        timeseries = df[target]
        result = adfuller(timeseries,autolag='AIC') #the adf function
        dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        print('Critical Value: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Lags used: %f' %result[2])
        print('Threshold Values:')
        for key, value in result[4].items(): #critical value dict
            print('%s: %.3f' % (key, value))
    adf_calc(df , 'Total_Sales')
    #Normalize the Dataset - Logarithmic transform
    df['total_sales_log'] = np.log(df.Total_Sales)
    df.head()
    # Defining a function to carry out the difference transform
    def differencing(df,date, order=1):
        df[date]=pd.to_datetime(df[date],errors='coerce') #setting date as datetime object and setting it as the index
        df=df.set_index(date)   
        print(f'The order of the transform is {order}')  
        if order == 1:
            #calculate t-1
            df['t_minus_1']=df['total_sales_log'].shift(periods=1)       
            #calculate first order dt
            df['target_order_dt']=df['total_sales_log'] - df['t_minus_1']       
        elif order == 2:
            #calculate t-2
            df['t_minus_1']=df['total_sales_log'].shift(periods=1)
            df['first_order_dt']=df['total_sales_log'] - df['t_minus_1'] #returns 1st order
            #have to shift the first order values then subtract that from the first order values
            df['first_order_dt_minus_1']=df['first_order_dt'].shift(periods=1)
            df['target_order_dt']=df['first_order_dt'] - df['first_order_dt_minus_1'] # returns 2nd order dt       
        elif order == 3:
            #calculate t=3 
            df['t_minus_1']=df['total_sales_log'].shift(periods=1)
            df['first_order_dt']=df['total_sales_log'] - df['t_minus_1'] # returns 1st order
            df['first_order_dt_minus_1']=df['first_order_dt'].shift(periods=1)
            df['second_order_dt']=df['first_order_dt'] - df['first_order_dt_minus_1'] # returns 2nd order
            df['second_order_dt_minus_1']=df['second_order_dt'].shift(periods=1)
            df['target_order_dt']=df['second_order_dt'] - df['second_order_dt_minus_1'] # returns 3rd order        
        else:
            raise Exception("Order for this particular problem shoud be <=3 ")
        #lets visualize the dataset after transforming it
        df[['target_order_dt']].plot(figsize=(15,8))
        plt.title(f'Total Sales After Differencing with Order {order} Against Time')
        plt.xlabel('Time')
        plt.ylabel("Total Sales After Differencing")
        plt.show()
        return df
    df_temp = df.reset_index()
    #1st order differencing
    df1_dt = differencing(df_temp, 'date', order=1) 
    display(df1_dt)
    #2nd order differencing
    df2_dt = differencing(df_temp, 'date', order=2)
    display(df2_dt)
    #3rd order differencing
    df3_dt = differencing(df_temp, 'date', order=3)
    display(df3_dt)
    #first order transformation of the log values for total sales looks more stationary than the original dataset.
    df2 = df1_dt.dropna()
    print("check stationarity of order 1")
    adf_calc(df2, 'target_order_dt') #Critical Value is greater than the Threshold values,consider the dataset as stationary.
    #Visualizing the Autocorrelation and the Partial autocorrelation curves
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df2.target_order_dt,lags=40,ax=ax1) #autocorrelation curve
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df2.target_order_dt,lags=40,ax=ax2) #partialautocorrelation curve
    #getting the metrics
    method_list=[] #going to hold the method
    algorithm_list = [] #specific algorithm
    rmse_list =[] # root mean squared error
    mse_list = [] #mean squared error
    mae_list = [] #mean absolute error
    r2_list = [] #correlation square -tells us how good does the data fit the line
    mape_list = [] #mean_absolute_percentage_error
    d = {
        'Method': method_list,
        'Algorithm':algorithm_list,
        'MSE': mse_list,
        'RMSE': rmse_list,
        'MAE': mae_list,
        'R2': r2_list,
        'MAPE': mape_list
    }
    def get_metric(actual,predicted,method,algorithm):
        mse = mean_squared_error(actual,predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual,predicted)
        r2 = r2_score(actual,predicted)
        mape = mean_absolute_percentage_error(actual,predicted)

        method_list.append(method)
        algorithm_list.append(algorithm)
        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
        mape_list.append(mape)
        result = pd.DataFrame(d)
        return result
    #forecasting with ARIMA   
    auto_arima_fit = pm.auto_arima(df.Total_Sales, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                             seasonal=False, trace=True, error_action='ignore', suppress_warnings=True,
                             stepwise=True)
    model = ARIMA(df.Total_Sales, order=(3,1,2))
    results = model.fit()
    print(results.summary())
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    # Prediction Plot
    def index_time(df, attribute):
        temp = df.reset_index()
        temp[attribute] = pd.to_datetime(temp[attribute],errors='coerce')
        df = temp.set_index(attribute)
        return df
    index_list=df.index.tolist()
    preds= results.predict(0,233)
    preds= preds.set_axis(index_list)
    preds=preds.to_frame()
    preds=index_time(preds,'index')#the pred values
    df = index_time(df,'date') #the actual values
    fig=matplotlib.pyplot.gcf() #plotting the current graph
    fig.set_size_inches(12.5,4.5)
    plt.title(f'Comparison between ARIMA and Actual Data')
    plt.xlabel('Time')
    plt.ylabel("Total Sales")
    plt.plot(df.Total_Sales[:234],label='Actual', color = 'black')
    plt.plot(preds,label = 'Predicted' , color = 'r')
    #Forecasting
    def get_forecast_values(df, N =30):
        last_date = df.reset_index().at[len(df)-1,'date'] #extracting the last date
        print(f"The last date is: {last_date}")
        forecast_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
        print(f"start forcast date: {forecast_date}")
        forecast = results.forecast(steps=N) #forecasting for N days 
        forecast_index = pd.date_range(start=forecast_date, periods=N)
        df_forecast = pd.DataFrame({'forecast':forecast.values},index=forecast_index)
        return df_forecast 
    #Forecasting the Total sales for the next 30 days using the ARIMA Model
    df_forecast = get_forecast_values(df,N = 30) 
    df_forecast
    #visualizing the forecasted values
    def forecast_plot(df, df_forecast, method):
        ci = df_forecast.values
        ax = df[100:].Total_Sales.plot(label='Past Sales', color ='black', figsize=(20, 15))
        df_forecast.plot(ax=ax,label='Forecasted Sales',color='darkred')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Sales')
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12.5, 4.5)
        plt.legend() 
        plt.title(f'Visualizing Forecasted Sales with {method}')
        plt.xlabel('Time')
        plt.ylabel("Total Sales")
        return plt.show()
    forecast_plot(df, df_forecast, 'ARIMA')
    #evaluating the model with metrics 
    ar_actual = df['Total_Sales'][:30] 
    ar_predicted = results.predict()[:30]
    ar_algorithm = 'ARIMA'
    ar_method = 'Time Series'
    x = get_metric(ar_actual,ar_predicted,ar_method,ar_algorithm)
    print(x)
    df_forecast.to_csv("forecasted_data.csv")
    df.to_csv("actual_data.csv")



