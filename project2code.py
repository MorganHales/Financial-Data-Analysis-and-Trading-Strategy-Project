# coding: utf-8

# In[47]:
import pandas as pd
import numpy as np
import warnings
import os
import yfinance as yf
import backtrader as bt
import quantstats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, precision_score
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from pyfolio import timeseries 
import zipfile

# In[2]:
def preprocess(file_path, date_column='date', date_format='%Y-%m-%d', closing_price_column='close', parse_dates=True):
    # Read the CSV file
    data_frame = pd.read_csv(file_path)

    # Parse dates if enabled
    if parse_dates:
        data_frame[date_column] = pd.to_datetime(data_frame[date_column], format=date_format, errors='coerce')

    # Sort the DataFrame by date
    data_frame.sort_values(by=date_column, inplace=True)

    # Create a target column indicating if the next day's close is greater than today's
    data_frame['target'] = (data_frame[closing_price_column].shift(-1) > data_frame[closing_price_column]).astype(int)

    # Save the processed DataFrame back to CSV
    data_frame.to_csv(file_path, index=False)

class CustomCSVData(bt.feeds.GenericCSVData):
    params = (
        ('custom_type', None),  # Custom type for different column settings
        ('dtformat', ('%Y-%m-%d')),  # Default datetime format
    )

    def __init__(self, *args, **kwargs):

        super(CustomCSVData, self).__init__(*args, **kwargs)

        if self.p.custom_type:
            self.set_columns()

    def set_columns(self):
        switcher = {
            'hourly_close': (-1, -1, -1, 2, -1),
            'daily_limit': (3, 4, 1, 2, 5),
            'daily_std': (2, 3, 1, 4, 5)
        }
        self.p.high, self.p.low, self.p.open, self.p.close, self.p.volume = \
            switcher.get(self.p.custom_type, (None, None, None, None, None))


# In[3]:
class new_features:
    def __init__(self, df, date_column='Date', close_column='Adj Close'): ## change if Adjusted Close not available
        self.df = df
        self.date_column = date_column
        self.close_column = close_column

    def calculate_moving_averages(self, window_list):
        for window in window_list:
            self.df[f'{window}d_MA'] = self.df[self.close_column].rolling(window=window).mean()

    def calculate_rsi(self):
        delta = self.df[self.close_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def calculate_macd(self):
        exp1 = self.df[self.close_column].ewm(span=12, adjust=False).mean()
        exp2 = self.df[self.close_column].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        #self.df['macd_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

    def process(self):
        self.df = self.df.reset_index()
        self.df = self.df.set_index(self.date_column)
        self.calculate_moving_averages([5, 25, 100])
        self.calculate_rsi()
        self.calculate_macd()

        # Extract columns
        moving_averages = self.df.filter(regex='_MA')
        rsi = self.df[['RSI']]
        macd = self.df[['MACD']]#, 'macd_signal'
        adj_close = self.df[[self.close_column]]
        
        # Concatenate all dataframes
        processed_df = pd.concat([moving_averages, rsi, macd, adj_close], axis=1)
        processed_df.dropna(subset = ['5d_MA', '25d_MA', '100d_MA','RSI',  'MACD'], inplace = True)
        return processed_df




# In[4]:

def test_accuracy(X_test: pd.DataFrame, y_test: np.ndarray, model) -> None:
    predictions = model.predict(X_test)
    print(predictions[:5])
    return accuracy_score(y_test, predictions)

def test_precision(X_test: pd.DataFrame, y_test: np.ndarray, model) -> None:
    predictions = model.predict(X_test)
    return precision_score(y_test, predictions)

def tune_hyperparams(model, model_type: str, x_train_set, y_train_set) -> None:
    if model_type == 'log_regression':
        param_grid = {
            'C': [ 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100] 
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose = 1)

    if model_type == 'rand_forest':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7]
        }
    
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, verbose=1)

    grid_search.fit(X_train, y_train)

    
    return grid_search.best_estimator_


# In[5]:


class MLStrategy(bt.Strategy):
    def __init__(self, model):
        # Load the pre-trained model
        self.model = model
        self.bar_executed = 0
        self.order = None
        self.longest_lookback = 100 ## current longest lookback period

        self.sma_5 = bt.indicators.SMA(self.data.close, period=5)
        self.sma_25 = bt.indicators.SMA(self.data.close, period=25)
        self.sma_100 = bt.indicators.SMA(self.data.close, period=100)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=12,  
                                       period_me2=26,  
                                       period_signal=9)

    def next(self):
        if len(self) < self.longest_lookback:
            return  # Not enough data yet for all indicators, skip this iteration
        indicators = self.get_model_input()
        prediction = self.model.predict(indicators)
        if prediction == 1 and not self.position:  
            self.log('Buy Create, %.2f' % self.data.close[0])
            self.order = self.buy(size=100) 
            
        elif prediction == 0 and self.position:  
            self.log('Sell Create, %.2f' % self.data.close[0])
            self.order = self.sell(size=100)  
            
        if self.position and len(self) >= (self.bar_executed + 4):
            self.log('Position Closed, %.2f' % self.data.close[0])
            self.order = self.close()
                
    def get_model_input(self):
        input_array = np.array([self.sma_5[0], self.sma_25[0], self.sma_100[0], self.rsi[0], self.macd[0]])
        input_array_2d = input_array.reshape(1, -1)
        return input_array_2d
    
    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f, Value: %.2f, Commission %.2f)" %
                    (order.executed.price, order.executed.value, order.executed.comm))
            else:
                self.log("Executed SELL (Price: %.2f, Value: %.2f, Commission %.2f)" %
                    (order.executed.price, order.executed.value, order.executed.comm))
            self.bar_executed = len(self) 
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None
        
    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        
# In[6]:


start_date = '2000-01-01'
end_date = '2021-11-12'


tickers_df = pd.read_csv('tickers.csv')
tickers = list(tickers_df['Ticker'])

train = yf.download(tickers, start = start_date, end = end_date) 

train.fillna(0, inplace = True)

train.ffill(inplace = True)


# In[7]:


train_close = train['Adj Close'].reset_index()
train_close = train_close.set_index('Date')

for col in train_close.columns:
    train_close[f'05d_{col}_moving_average'] = train_close[col].rolling(window=5).mean()
    train_close[f'25d_{col}_moving_average'] = train_close[col].rolling(window=25).mean()
    train_close[f'100d_{col}_moving_average'] = train_close[col].rolling(window=100).mean()

    # Calculate RSI
    delta = train_close[col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    train_close[f'{col}_rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = train_close[col].ewm(span=12, adjust=False).mean()
    exp2 = train_close[col].ewm(span=26, adjust=False).mean()
    train_close[f'{col}_macd'] = exp1 - exp2
    train_close[f'{col}_macd_signal'] = train_close[f'{col}_macd'].ewm(span=9, adjust=False).mean()


average_5_cols = [col for col in train_close.columns if '05d' in col]
average_25_cols = [col for col in train_close.columns if '25d' in col]
average_100_cols = [col for col in train_close.columns if '100d' in col]
rsi_cols = [col for col in train_close.columns if '_rsi' in col]
macd_cols = [col for col in train_close.columns if '_macd' in col or '_macd_signal' in col]
close_cols = [col for col in train_close.columns if 'average' not in col and '_rsi' not in col and '_macd' not in col]

averages_5 = pd.concat([train_close[col] for col in average_5_cols], ignore_index=True)
averages_25 = pd.concat([train_close[col] for col in average_25_cols], ignore_index=True)
averages_100 = pd.concat([train_close[col] for col in average_100_cols], ignore_index=True)
rsi_values = pd.concat([train_close[col] for col in rsi_cols], ignore_index=True)
macd_values = pd.concat([train_close[col] for col in macd_cols], ignore_index=True)
closes = pd.concat([train_close[col] for col in close_cols], ignore_index=True)

train_df = pd.concat([averages_5, averages_25, averages_100, rsi_values, macd_values, closes], 
                     axis=1, keys=['5d_MA', '25d_MA', '100d_MA', 'RSI', 'MACD', 'Adj_Close'])


# In[8]:


train_df = train_df.dropna(subset = ['5d_MA', '25d_MA', '100d_MA','RSI',  'MACD', 'Adj_Close'])#'MACD' ,
train_df['target'] = (train_df['Adj_Close'].shift(-1) > train_df['Adj_Close']).astype(int)


# In[9]:
X = train_df[[x for x in train_df.columns if 'target' not in x and 'Close' not in x]]
y = train_df['target']
# In[10]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_clf = RandomForestClassifier()  #
rf_model_tuned = tune_hyperparams(rf_clf, 'rand_forest', X_train_scaled, y_train)


LR_model = LogisticRegression()
LR_model_tuned = tune_hyperparams(LR_model, 'log_regression', X_train_scaled, y_train)



# In[57]:
LR_pred = LR_model_tuned.predict(X_test_scaled)
RF_pred = rf_model_tuned.predict(X_test_scaled)

# Evaluating the model
LR_accuracy = accuracy_score(y_test, LR_pred)
rf_accuracy = accuracy_score(y_test, RF_pred)

LR_precision = precision_score(y_test, LR_pred)
rf_precision = precision_score(y_test, RF_pred)


# In[12]:
zip_file_path = 'stock_dfs.zip'
csv_files = []

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    all_files = zip_ref.namelist()    
    csv_files = [file.replace('.csv', '').replace('stock_dfs/', '') for file in all_files if file.endswith('.csv')]

oos_testing1 = pd.read_csv('tickers_nasd.csv', usecols=['Symbol'])
oos_testing2 = pd.read_csv('tickers_nyse.csv', usecols = ['Symbol'])
oos_testing = pd.concat([oos_testing1, oos_testing2], axis =0 )

oos_testing_list = list(set(oos_testing['Symbol']) & set(csv_files))

oos_lr_testing = pd.DataFrame(columns=['Ticker', 'Precision', 'Accuracy'])
oos_rf_testing = pd.DataFrame(columns=['Ticker', 'Precision', 'Accuracy'])


# In[14]:
zip_path = './stock_dfs.zip'# zip_path needs to be defined, e.g., zip_path = 'path/to/your/zipfile.zip'

extracted_files_dir = 'extracted_csv_files'
os.makedirs(extracted_files_dir, exist_ok=True)

for ticker in oos_testing_list:
    file_name = f"stock_dfs/{ticker}.csv"
    extracted_file_path = os.path.join(extracted_files_dir, file_name)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if file_name in zip_ref.namelist():
            zip_ref.extract(file_name, path=extracted_files_dir)
            
            preprocess(extracted_file_path, parse_dates = True)

            data = pd.read_csv(extracted_file_path)#CustomCSVData(dataname=extracted_file_path, custom_type='daily_std')

            feature_eng = new_features(data, date_column = 'date', close_column = 'close')
            features =  feature_eng.process()
            target = (features['close'].shift(-1) > features['close']).astype(int)
            if len(features) < 100:
                continue

            X_test = features.drop('close', axis = 1)
            y_test = target 

            LR_pred = LR_model_tuned.predict(X_test)
            RF_pred = rf_model_tuned.predict(X_test)

            LR_precision = precision_score(y_test, LR_pred, average='binary')
            LR_accuracy = accuracy_score(y_test, LR_pred)
            RF_precision = precision_score(y_test, RF_pred, average='binary')
            RF_accuracy = accuracy_score(y_test, RF_pred)

            oos_lr_testing = pd.concat([oos_lr_testing, pd.DataFrame({'Ticker': [ticker], 'Precision': [LR_precision], 'Accuracy': [LR_accuracy]})], ignore_index=True)
            oos_rf_testing = pd.concat([oos_rf_testing, pd.DataFrame({'Ticker': [ticker], 'Precision': [RF_precision], 'Accuracy': [RF_accuracy]})], ignore_index=True)

            
        else:
            print(f"{file_name} not found in zip archive.")


# In[56]:
top_10_lr_stocks = oos_lr_testing.sort_values(by='Accuracy', ascending=False).head(10)
top_10_rf_stocks = oos_rf_testing.sort_values(by='Accuracy', ascending=False).head(10)

# In[40]:
trading_report_stocks = list(set(top_10_rf_stocks['Ticker']) | set(top_10_lr_stocks['Ticker']))

oos_lr_results = {}
oos_rf_results = {}

# In[49]:
report_storage = {}
perf_stats_storage = {}

# Create the 'reports' directory if it does not exist
reports_dir = 'reports'
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

for ticker in trading_report_stocks:
    
    path = f"extracted_csv_files/stock_dfs/{ticker}.csv"
    data = CustomCSVData(dataname = path, custom_type = 'daily_std')

    for model_name, model in [('LR', LR_model_tuned), ('RF', rf_model_tuned)]:
        cerebro = bt.Cerebro()
        ##add pyfolio observer
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        
        cerebro.addstrategy(MLStrategy, model=model)
        cerebro.broker.setcash(100000)
        cerebro.broker.setcommission(commission=0.001)
        
        try:
            cerebro.adddata(data)
            results = cerebro.run()
            final_value = cerebro.broker.getvalue()
            
            strat = results[0]
            portfolio_stats = strat.analyzers.getbyname('pyfolio')
            returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
            returns.index = returns.index.tz_convert(None)

            # Generate and save QuantStats report for each ticker and model
            report_filename = f'reports/{ticker}_{model_name}_report.html'
            quantstats.reports.html(returns, output=report_filename, title=f'BT Analysis - {ticker} - {model_name}')
            
            
            report_storage[ticker] = {
                'returns': returns,
                'positions': positions,
                'transactions': transactions,
                'gross_lev': gross_lev
            }
            perf_func = timeseries.perf_stats 
            perf_stats_all = perf_func( returns= returns, positions=positions, transactions=transactions)
            perf_stats_storage[ticker] = perf_stats_all

            
            
        except Exception as e:
            print(f"Error running strategy for {ticker} with {model_name}: {e}")
            continue
        
        if model_name == 'LR':
            oos_lr_results[ticker] = final_value
        else:
            oos_rf_results[ticker] = final_value
        
        print(f"<{model_name}> Strategy Final Balance for {ticker}: ${final_value:.2f}")


# In[54]:
sharpe_ranking = sorted(perf_stats_storage, key=lambda x: perf_stats_storage[x]['Sharpe ratio'], reverse=True)


max_drawdown_ranking = sorted(perf_stats_storage, key=lambda x: perf_stats_storage[x]['Max drawdown'])


print("Ranking by Sharpe Ratio:")
for stock in sharpe_ranking:
    print(f"{stock}: {perf_stats_storage[stock]['Sharpe ratio']}")

print("\nRanking by Max Drawdown:")
for stock in max_drawdown_ranking:
    print(f"{stock}: {perf_stats_storage[stock]['Max drawdown']}")


# In[59]:
df = pd.DataFrame(perf_stats_storage).transpose()

df.to_csv('output.csv')