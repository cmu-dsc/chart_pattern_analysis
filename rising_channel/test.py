import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.signal import find_peaks, argrelmin
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os, random

sys.path.append('C:/Users/OuHuiyi/Desktop/Stock pattern analysis/python_utils')
sys.path.append('C:/Users/OuHuiyi/Desktop/Stock pattern analysis/chart_pattern_analysis')
from python_utils import time_utils, file_utils, list_utils, math_utils
# from chart_pattern_analysis.real_time_trading_master import get_init_tasks

# ticker for reference
symbols = ['AGN', 'JBHT', 'CPB', 'EVHC', 'IDXX', 'QRVO', 'JWN', 'SBAC', 'EOG', 'TAP', 'VRTX', 'BWA', 'UAL',
           'UAA', 'AAPL', 'SPG', 'FIS', 'GT', 'GIS', 'PYPL', 'WRK', 'ROP', 'GE', 'GD', 'VAR', 'GM', 'ALK',
           'MAS', 'MAR', 'MAT', 'SNA', 'FOXA', 'SNI', 'MAA', 'MAC', 'SIG', 'XYL', 'TSN', 'SYMC', 'BEN',
           'CMI', 'EL', 'CMG', 'ALB', 'CME', 'CMA', 'NKE', 'CMS', 'PCG', 'VLO', 'HUM', 'PCLN', 'GRMN', 'FTV',
           'SNPS', 'K', 'FFIV', 'BLL', 'HCA', 'BLK', 'FTI', 'HCN', 'NFX', 'MA', 'CBG', 'PRGO', 'MO', 'HLT',
           'NOC', 'MU', 'CBS', 'NDAQ', 'UHS', 'MS', 'TJX', 'NOV', 'AMGN', 'FB', 'COST', 'FE', 'CTL', 'DPS',
           'FL', 'REG', 'MSFT', 'GLW', 'SCHW', 'BSX', 'GGP', 'FCX', 'BDX', 'WHR', 'BXP', 'MDT', 'F', 'DFS',
           'JEC', 'ILMN', 'V', 'FMC', 'MPC', 'TSCO', 'ALL', 'NTAP', 'LMT', 'MMM', 'SO', 'JNPR', 'MMC', 'CTSH',
           'ADBE', 'IBM', 'BAX', 'CAT', 'CAH', 'BAC', 'GS', 'CAG', 'LB', 'INCY', 'LH', 'PGR', 'HIG', 'CELG',
           'WLTW', 'A', 'ABC', 'ZION', 'STI', 'STT', 'ABT', 'XOM', 'STX', 'STZ', 'UDR', 'CERN', 'RSG',
           'PNW', 'BHF', 'IVZ', 'PNC', 'GOOGL', 'CBOE', 'VFC', 'AMZN', 'AON', 'PRU', 'RE', 'RF', 'AOS',
           'RL', 'CHD', 'CHK', 'L', 'HRS', 'HRL', 'MRO', 'TPR', 'MRK', 'HRB', 'IPG', 'NLSN', 'RHT', 'RHI',
           'APH', 'ROST', 'APC', 'APA', 'APD', 'KSU', 'AEE', 'WFC', 'IFF', 'CHTR', 'NVDA', 'KORS', 'XRAY',
           'AVGO', 'RTN', 'PKG', 'BMY', 'AFL', 'DISCA', 'PAYX', 'OXY', 'SHW', 'ED', 'DISCK', 'EA', 'ORLY',
           'DLR', 'CSX', 'JNJ', 'EW', 'ES', 'ANTM', 'MGM', 'ALGN', 'LUV', 'WYN', 'LUK', 'ADSK', 'LNT', 'ECL', 'SEE', 'MTD', 'RMD', 'PVH', 'DGX', 'NUE', 'LNC', 'EXPD', 'UNP', 'ETFC', 'DUK', 'BHGE', 'VTR', 'CLX', 'DTE', 'BBY', 'FRT', 'UNH', 'BBT', 'TRIP', 'INFO', 'KO', 'VMC', 'ARNC', 'KR', 'EXPE', 'KEY', 'ITW', 'KLAC', 'SWK', 'PNR', 'WYNN', 'AAL', 'ANSS', 'AAP', 'WBA', 'XL', 'TGT', 'HBAN', 'BIIB', 'TSS', 'ATVI', 'AKAM', 'INTC', 'HOLX', 'M', 'MLM', 'INTU', 'ALXN', 'SLG', 'DHR', 'YUM', 'AJG', 'MCO', 'JPM', 'MCD', 'NRG', 'PBCT', 'AXP', 'NFLX', 'EXC', 'WM', 'SRCL', 'WU', 'EXR', 'WY', 'VNO', 'CDNS', 'UNM', 'AWK', 'HST', 'HSY', 'DLPH', 'FBHS', 'WEC', 'OMC', 'GPN', 'HAL', 'FITB', 'EQR', 'EQT', 'GPC', 'XLNX', 'HAS', 'CTXS', 'GPS', 'SYF', 'PHM', 'SYK', 'LEG', 'KIM', 'SYY', 'AEP', 'AES', 'AET', 'EMN', 'HCP', 'ESRX', 'AMAT', 'SWKS', 'CTAS', 'EMR', 'FDX', 'CRM', 'NWSA', 'PX', 'PG', 'PH', 'PM', 'EFX', 'C', 'MHK', 'FOX', 'COTY', 'QCOM', 'XRX', 'ULTA', 'MOS', 'PSA', 'PSX', 'MON', 'MCHP', 'ETR', 'COP', 'DVN', 'KMB', 'PEP', 'KMI', 'GWW', 'FISV', 'COG', 'COF', 'PEG', 'ETN', 'COL', 'COO', 'KMX', 'DLTR', 'ZBH', 'DG', 'ANDV', 'CI', 'DE', 'HES', 'CB', 'CA', 'CF', 'WAT', 'ZTS', 'CHRW', 'TDG', 'SBUX', 'PLD', 'CL', 'CMCSA', 'MKC', 'PDCO', 'CVX', 'AIZ', 'EIX', 'AIV', 'CVS', 'RCL', 'LOW', 'AYI', 'AIG', 'DIS', 'PPG', 'MNST', 'NSC', 'PPL', 'OKE', 'GOOG', 'IP', 'HPQ', 'IR', 'IT', 'URI', 'IRM', 'MTB', 'UTX', 'JCI', 'HPE', 'PWR', 'CSCO', 'RJF', 'ARE', 'WDC', 'HBI', 'BA', 'BK', 'DRI', 'CCL', 'CCI', 'TIF', 'DRE', 'ALLE', 'FLIR', 'MYL', 'LYB', 'KHC', 'HSIC', 'D', 'NAVI', 'DVA', 'BF.B', 'WMT', 'VZ', 'T', 'WMB', 'VRSN', 'VRSK', 'CINF', 'NTRS', 'CXO', 'HP', 'FLR', 'REGN', 'HD', 'LKQ', 'LLY', 'TMK', 'DHI', 'BRK.B', 'NWS', 'DOV', 'SCG', 'AME', 'AMD', 'AMG', 'LLL', 'NWL', 'AMP', 'TMO', 'AMT', 'LRCX', 'O', 'UPS', 'TRV', 'AVY', 'ABBV', 'DWDP', 'MDLZ', 'UA', 'AVB', 'NEM', 'NEE', 'ACN', 'NI', 'TEL', 'RRC', 'FAST', 'SLB', 'NBL', 'FLS', 'ADS', 'ADP', 'GILD', 'SJM', 'MCK', 'ADM', 'ADI', 'VIAB', 'XEC', 'XEL', 'LEN', 'EBAY', 'MET', 'USB', 'AZO', 'MSI', 'DISH', 'TWX', 'PCAR', 'TXN', 'DAL', 'SRE', 'ORCL', 'TXT', 'PXD', 'HOG', 'ESS', 'PKI', 'CNP', 'HON', 'ICE', 'CSRA', 'ROK', 'CNC', 'TROW', 'ISRG', 'PFE', 'EQIX', 'PFG', 'SPGI', 'DXC']


def find_matching_charts(startP, peak, bottom, rollAvg, params, tib, i, method=4, p=0.5):
    potential_buy = False
    if method == 4:
        period = params[0]
        numAdjacent, lag, tol, enterP, targetP, stpLossP = params[2:]
        print('params', params)
        # find the high and low, store the time index
        if rollingAvg[i // 2 + 1] == rollingAvg[(i - 2 * numAdjacent) : (i+1)].max():
            peak.append(i // 2 + 1)
        elif rollingAvg[i // 2 + 1] == rollingAvg[(i - 2 * numAdjacent) : (i+1)].min():
            bottom.append(i // 2 + 1)

        # select the two highest high and two lowest lows, should get the corresponding prices
        if len(peak) > 1:
            peak.sort(reverse=True)
            peak = peak[:2]
            k_resis = (rollingAvg[peak[0]] - rollingAvg[peak[1]]) / (peak[1] - peak[0])
        if len(bottom) > 1:
            bottom.sort()
            bottom = bottom[:2]
            k_support = (rollingAvg[bottom[1]] - rollingAvg[bottom[0]])


    return potential_buy


def vwap(data):
    volume = data.volume
    price = data.price
    return data.assign(vwap = np.dot(volume, price) / volume.sum())


# load and preprocess data
def load_data(path, symbol):
    data_dir = ''
    today_single_data_file = np.load(data_dir + '/1_18_2018.npz')
    index = pd.date_range('1/18/2018 09:30:00', '1/18/2018 16:00:00', freq= 'S')
    today_price_file = pd.DataFrame(today_single_data_file[symbols[symbol]],
                                    index = index[:len(today_single_data_file[symbols[symbols]][:,0])],
                                    columns= ['price', 'volume'])
    # convert cumulative volume to get vwap, resample
    today_price_file['volume'] = today_price_file['volume'].diff(periods=1).dropna()
    today_price_file = today_price_file.groupby(today_price_file.index.date, group_keys=False).apply(vwap)
    aggAvg = today_price_file.resample('30S').mean()
    aggAvg.to_csv('test_data.csv')
    return


def reformat_to_series(data_to_reformat, original_data, index_name='date', old_index_name='timestamp', new_index_name='day_num', column_name='close'):
    data_to_reformat.index.name = index_name
    data_to_reformat = data_to_reformat.reset_index()
    data_to_reformat = data_to_reformat[~data_to_reformat[index_name].duplicated()]
    p = original_data.reset_index()
    data_to_reformat[new_index_name] = p[p[old_index_name].isin(data_to_reformat[index_name])].index.values
    data_to_reformat = data_to_reformat.set_index(new_index_name)[column_name]
    return data_to_reformat


def plot_minmax_pattern(aggAvg, local_maxima, local_minima):
    plt.figure()
    ax = plt.gca()
    plt.plot(aggAvg.price, color='orange', linewidth=0.8, label=symbols[1] + ' ' + 'price')
    plt.scatter(local_maxima.index, local_maxima.price, s=10, color='b',
                label=symbols[symbol] + ' ' + 'local maxima')
    plt.scatter(local_minima.index, local_minima.price, s=10, color='r',
                label=symbols[symbol] + ' ' + 'local minima')
    plt.legend(loc='best', frameon=False)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.show()
    return

# path = 'C:/Users/OuHuiyi/Desktop/Stock pattern analysis/rising channel/npz'
symbol = 1
# load_data(path,symbol)
tib = 7
# 8 params: [period, rolling_mean_interval, num_adjacent, lag, tolerance, enterP, targetP, stpLossP]
params = pd.read_csv('method_4_params_8.csv', header = None,
                     names=['period','window','num_adjacent','num_lag','tol','enterP','targetP','stpLossP'])
aggAvg = pd.read_csv('test_data.csv', header = 0, index_col=0)

# plot min and max, x-axis doesn't appear
maxima, _ = find_peaks(aggAvg.price, distance= params['num_adjacent'][0], threshold= 0.02)
local_maxima = aggAvg.iloc[maxima,:]
# local_maxima = reformat_to_series(local_maxima, smooth_prices, index_name='date', old_index_name='timestamp',
#                                   new_index_name='day_num', column_name='close')
minima = argrelmin(aggAvg.price.values, order = params['num_adjacent'][0], mode='clip')[0]
local_minima = aggAvg.iloc[minima,:]
# plot_minmax_pattern(aggAvg, local_maxima, local_minima)

# find pattern




