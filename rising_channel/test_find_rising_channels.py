# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 00:34:10 2019

@author: timcr
"""

import pandas as pd
import numpy as np
import datetime as dt
import sys, os
import matplotlib as mpl
plt = mpl.pyplot
from scipy.signal import argrelextrema
from copy import deepcopy
import time
#sys.path.append(os.path.join(os.environ['HOME'], 'python_utils'))
#import file_utils
from chart_patterns.pattern_detection import find_channels

def vwap(data):
    price = data['Mark']
    diff_vol = data['Volume']
    if sum(diff_vol) == 0:
        entry = np.array([[price[0], sum(diff_vol)]])
    else:
        entry = np.array([[np.dot(price, diff_vol) / sum(diff_vol), sum(diff_vol)]])
    return pd.DataFrame(entry, columns=['vwap', 'vol'])

def to_timestamp_index(df):
    return df.index.values.astype(np.int64) // 10 ** 9

def main():
    fpath = os.path.join('npz_filled', '11_8_2018.npz')
    num_seconds_resample_period = 300
    d = dt.datetime(2018, 11, 8, 9, 30)
    d_shifted = d + dt.timedelta(seconds=num_seconds_resample_period)
    
    num_seconds_in_trade_day = int(3600 * 6.5)
    num_periods = num_seconds_in_trade_day // num_seconds_resample_period
    #csv_data = np.array(file_utils.read_csv(fpath), dtype='float32')
    npz_data = np.load(fpath)
    it = 0
    for sym in npz_data:

        sym_data = npz_data[sym]
    
        df = pd.DataFrame(sym_data, columns=['Mark','Volume'], index=pd.date_range(d.strftime('%m/%d/%Y %H:%M:%S'), periods=num_seconds_in_trade_day, freq= '1S'))
        df['Volume'] = df['Volume'].diff(periods=1).fillna(0)
        resampled_df = df.resample(str(num_seconds_resample_period) + 'S').apply(vwap)
        resampled_df.set_index(pd.date_range(d_shifted.strftime('%m/%d/%Y %H:%M:%S'), periods=num_periods, freq=str(num_seconds_resample_period) + 'S'), inplace=True)
        
        local_max = argrelextrema(resampled_df['vwap'].values, np.greater)[0]
        local_min = argrelextrema(resampled_df['vwap'].values, np.less)[0]
        maxes = resampled_df['vwap'].iloc[local_max]
        mins = resampled_df['vwap'].iloc[local_min]
        
        resampled_df.index = to_timestamp_index(resampled_df)
        maxes.index = to_timestamp_index(maxes)
        mins.index = to_timestamp_index(mins)
        
        numeric_maxes = np.vstack([maxes.index.values, maxes.values]).T
        numeric_mins = np.vstack([mins.index.values, mins.values]).T
        
        tol = 0.000000001
        poke_out_tol = 0.05
        channel_max_tails, channel_min_tails, channel_max_heads, channel_min_heads, smaxtail, smintail, smaxhead, sminhead = find_channels(numeric_maxes, numeric_mins, tol, poke_out_tol)
        
        number_of_channels = len(channel_max_tails)
        
        if number_of_channels == 0:
            continue
        print(sym)
        for i in range(number_of_channels):
           lines = plt.plot(np.vstack([channel_max_tails[i,0], channel_max_heads[i,0]]), 
                    np.vstack([channel_max_tails[i,1], channel_max_heads[i,1]]))
           plt.plot(np.vstack([channel_min_tails[i,0], channel_min_heads[i,0]]),
                    np.vstack([channel_min_tails[i,1], channel_min_heads[i,1]]), c=lines[0].get_color())
           
           plt.plot(np.vstack([smaxtail[i,0], smaxhead[i,0]]),
                    np.vstack([smaxtail[i,1], smaxhead[i,1]]), c=lines[0].get_color())
           plt.plot(np.vstack([smintail[i,0], sminhead[i,0]]),
                    np.vstack([smintail[i,1], sminhead[i,1]]), c=lines[0].get_color())
        
        plt.plot(resampled_df.index, resampled_df['vwap'], c='black')
        plt.scatter(maxes.index, maxes)
        plt.scatter(mins.index, mins)
        plt.show()
        it += 1
        #if it > 100:
        #    break

if __name__ == "__main__":
    main()