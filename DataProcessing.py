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

            # TODO pos, neg cộng dồn 14 ngày trước

            avg_gain = 0
            avg_loss = 0

            for i in range(1,self.period):
                if(diff[i]>0):
                    avg_gain+=diff[i]
                else:
                    avg_loss+=diff[i]
            
            avg_gain/=self.period
            avg_loss/=self.period

            for i in range(self.period,len(diff)):
                pos = 0
                neg = 0
                for j in range(i-self.period+1,i+1):
                    day_data = df.iloc[j]
                    if(diff[j]<0):
                        avg_loss = (avg_loss*(self.period-1)-diff[j])/self.period
                        neg += (day_data['High']+day_data['Low']+day_data['Close'])/3 * day_data['Volume']
                    else:
                        avg_gain = (avg_gain*(self.period-1)+diff[j])/self.period
                        pos += (day_data['High']+day_data['Low']+day_data['Close'])/3 * day_data['Volume']
                neg -= mfi[j-self.period]
                pos -= mfi[j-self.period]
                
                rsi.append(100-100/(1+avg_gain/avg_loss))
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
                ema_12 = (df['Close'].iloc[i]-ema_prev)*0.1538 + ema_prev
                ema_26 = 0
                if i > 26:
                    ema_prev = np.mean(ema[i-26-1:i-1])
                    ema_26 = (df['Close'].iloc[i]-ema_prev)*0.1538 + ema_prev
                macd.append(ema_12-ema_26)

            # Insert indicators into the data
            indicators = {
                'RSI':rsi,
                'MFI':mfi,
                'EMA':ema,
                'SO':so,
                'MACD':macd
            }

            processed_data = df.copy()
            processed_data = processed_data.assign(**indicators)

            # Insert next day price into the data
            next_day_price = []
            for i in range(len(diff)-1):
                next_day_price.append(df['Close'].iloc[i+1])
            next_day_price.append(0)

            processed_data.insert(len(processed_data.columns),'Next Day Price',next_day_price)
            # processed_data = processed_data.drop(['Open','High','Low'],axis=1)
            processed_data = processed_data.drop(['Open'],axis=1)
            processed_data = processed_data.iloc[self.period:len(processed_data)-1]

            # Save data
            path = f'data/processed/{ticker}.csv'
            processed_data.to_csv(path,index=False)
            print(f'Saved {ticker} data at data/processed/{ticker}.csv')
    
    def get_index_names(self)->list:
        print(f'S&P 500: {self.SP500_tickers}\n VN: {self.VN_tickers}')
        return self.tickers

    def get_data(self,index_name)->pd.DataFrame:
        try:
            df = pd.read_csv(f'data/processed/{index_name}.csv')
            return df
        except:
            print(f'Data of {index_name} not existed/processed, try Data.preprocess({index_name})')
            return None
    