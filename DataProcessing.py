import pandas as pd
import numpy as np

class Data:
    def __init__(self,period = 14):
        self.SP500_tickers = ["AAPL", "ORCL", "GOOGL", "AMZN", "BK", "LIFE", "T", "XOM"]
        self.VN_tickers = ["VHM", "CTR", "FPT", "MBB"]
        self.tickers = self.SP500_tickers + self.VN_tickers
        self.period = period

    def preprocess(self,*args)->None:
        if(args==()):
            tickers = self.tickers
        else: 
            tickers = args

        for ticker in tickers:
            try:
                df = pd.read_csv(f'data/raw/{ticker}.csv')
            except:
                print(f'Data of {ticker} not existed')
                return None

            print(f'Processing {ticker}.csv')
            df = df.drop(['Unnamed: 0','Dividends','Stock Splits'],axis=1)


            # Calculate difference: diff[date] = Close[date] - Close[date-1]
            # diff[0] = 0
            diff = []
            diff.append(0)
            for day in range(1,len(df['Close'])):
                diff.append(df['Close'].iloc[day]-df['Close'].iloc[day-1])

            # Calculate RSI, MFI, SO
            rsi = []
            mfi = []
            so = []

            # First period in data has no information about the previous period so RSI = MFI = SO = 0
            for i in range(self.period):
                rsi.append(0)
                mfi.append(0)
                so.append(0)


            for i in range(self.period,len(diff)):
                gain = 0
                loss = 0
                pos = 0
                neg = 0
                for j in range(i-self.period+1,i+1):
                    day_data = df.iloc[j]
                    if(diff[j]<0):
                        loss+=diff[j]
                        neg += (day_data['High']+day_data['Low']+day_data['Close'])/3 * day_data['Volume']
                    else:
                        gain+=diff[j]
                        pos += (day_data['High']+day_data['Low']+day_data['Close'])/3 * day_data['Volume']

                rsi.append(100-100/(1+gain/-loss))
                mfi.append(100-(100/(1+pos/neg)))
                
                highest = -np.inf
                lowest = np.inf
                for j in range(i-self.period+1,i+1):
                    day_data = df.iloc[j]
                    if(day_data['Low']<lowest):
                        lowest = day_data['Low']
                    if(day_data['High']>highest):
                        highest = day_data['High']
                so.append(((df['Close'].iloc[i]-lowest)/(highest-lowest))*100)

            # Calculate EMA
            alpha = 2/(self.period+1)
            ema = [df['Close'].iloc[0]]
            
            for i in range(1,len(diff)):
                ema.append(alpha*df['Close'].iloc[i] + (1-alpha)*ema[i-1])

            # Calculate MACD
            macd = []
            for i in range(self.period):
                macd.append(0)

            for i in range(self.period,len(diff)):
                ema_prev = np.mean(ema[i-12-1:i-1])
                macd.append((df['Close'].iloc[i]-ema_prev)*0.15 + ema_prev)

            # Insert indicators into the data
            indicators = {
                'RSI':rsi,
                'MFI':mfi,
                'EMA':ema,
                'SO':so,
                'MACD':macd
            }

            proccessed_data = df.copy()
            proccessed_data = proccessed_data.assign(**indicators)

            # Insert next day price into the data
            next_day_price = []
            for i in range(len(diff)-1):
                next_day_price.append(df['Close'].iloc[i+1])
            next_day_price.append(0)

            proccessed_data.insert(len(proccessed_data.columns),'Next Day Price',next_day_price)
            proccessed_data = proccessed_data.drop(['Open','High','Low'],axis=1)
            proccessed_data = proccessed_data.iloc[self.period:len(proccessed_data)-1]

            # Save data
            path = f'data/proccessed/{ticker}.csv'
            proccessed_data.to_csv()
            proccessed_data.to_csv(path,index=False)
            print(f'Saved {ticker} data at data/proccessed/{ticker}.csv')
    
    def get_index_names(self)->list:
        print(f'S&P 500: {self.SP500_tickers}\n VN: {self.VN_tickers}')
        return self.tickers

    def get_data(self,index_name)->pd.DataFrame:
        try:
            df = pd.read_csv(f'data/proccessed/{index_name}.csv')
            return df
        except:
            print(f'Data of {index_name} not existed/proccessed, try Data.preprocess({index_name})')
            return None
    