import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.environ['HOME'], 'python_utils'))
import file_utils

def vwap(data):
    print('data')
    print(data)
    print('end data')
    price = data['a']
    diff_vol = data['b']
    
    return pd.DataFrame(np.array([[np.dot(price, diff_vol) / sum(diff_vol), sum(diff_vol)]]), columns=['q', 'w'])

def main():
    fpath = 'test_resample_data.csv'
    csv_data = np.array(file_utils.read_csv(fpath), dtype='float32')

    df = pd.DataFrame(csv_data, columns=['a','b'], index=pd.date_range('1/18/2018 09:30:00', periods=9, freq= '1S'))
    df['b'] = df['b'].diff(periods=1).fillna(0)
    resampled_df = df.resample('3S').apply(vwap)
    resampled_df.set_index(pd.date_range('1/18/2018 09:30:02', periods=3, freq= '3S'), inplace=True)
    print(resampled_df)

if __name__ == "__main__":
    main()