# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:14:05 2018

@author: timcrose
"""
import os, sys, random
#sys.path.append(r'E:\Python')
sys.path.append('/export/home/math/trose/python_utils')
import math_utils, time_utils
import file_utils

def get_upper_vol_to_try_for(sym_vol_list, p, t=1, go_back=280):
    '''
    sym_vol_list: list of numbers
        List where each element is the cumulative vol of a stock for 
        today up until the current time. There should be one entry
        per second since the opening bell of today.
    p: float
        0 < lbd < p <= 1. lbd is a lower bound such that as long as
        p is this low, the order shouldn't have much trouble going 
        through. p * (avg volume increase in t second-intervals) 
        will be the volume target upper bound.
    t: float
        p * (avg volume increase in t second-intervals)
        will be the volume target upper bound.
    go_back: int
        Number of seconds from the current time to use for getting
        volume estimate. If len(sym_vol_list) < go_back, then 
        go_back = len(sym_vol_list).
        
    Return: float
        Upper bound on volume to try for. This volume may be higher than
        the amount of buying power available; if this is the case then
        the buying power will limit the volume to try for and will likely
        use the remainder of the buying power with the next trade.
    Purpose: As part of approximating whether an order will actually execute,
        we need to not try for an order with too high of a volume: orders with
        volumes higher than the average volume increases in t seconds are possibly
        unlikely to execute in less than t seconds.
    '''
    if len(sym_vol_list) < go_back:
        go_back = len(sym_vol_list)
    sym_vol_list = sym_vol_list[-go_back:]
    vol_t_list = []
    i = 0
    while i + t < len(sym_vol_list):
        vol_t_list.append(sym_vol_list[i + t] - sym_vol_list[i])
        i += t
    if len(vol_t_list) == 0:
        avg_vol_incr_in_t_secs = 0
    else:
        avg_vol_incr_in_t_secs = math_utils.mean(vol_t_list)
    return int(p * avg_vol_incr_in_t_secs)

def calc_buying_power_needed(approx_num_shares, target_buy_price):
    buying_power_wanted = approx_num_shares * target_buy_price
    # $0.005 per share is one method IB calculates commission
    # 0.5% of the sale value is the other method IB uses to calculate commission
    # The option chosen is whichever is cheaper.
    lower_limit = round(approx_num_shares * 0.005, 3)
    upper_limit = round(buying_power_wanted * 0.005, 3)
    if lower_limit < upper_limit:
        commission = lower_limit
    else:
        commission = upper_limit

    if commission < 1.0:
        commission = 1.0

    #Only commission for the trade entering a position is accounted for here because
    # that's what you need in order to enter a trade.

    buying_power_needed = buying_power_wanted + commission
    return buying_power_needed, commission

def calc_vol_to_try_for(buying_power_wanted, target_buy_price, sym_vol_list, p, t=1.0, go_back=280):
    vol_upbd = get_upper_vol_to_try_for(sym_vol_list, p, t, go_back)
    
    vol_buy_power_bd = int(buying_power_wanted / target_buy_price)
    
    if vol_upbd > vol_buy_power_bd:
        return vol_buy_power_bd
    else:
        return vol_upbd

def find_matching_charts(dct, params, buy_today, max_buying_power_available, buying_power_available, tib, i, todays_data_file, rank, method=3, p=0.5):
    if method == 3:
        pop_pct = params[0]
        hop_pct = params[1]
        stop_pct = params[2]
        print('params', params)
        buying_power_wanted = round(max_buying_power_available / params[3], 3)
        for sym in dct:
            #if haven't bought this symbol before today:

            sym_found = False
            for row in buy_today:
                if sym == row[0]:
                    sym_found = True
            if sym_found:
                continue
            #if stock meets buy criteria
            dop = dct[sym][0,0]
            if dop != 0:
                try:
                    target_buy_price = math_utils.round_nearest_multiple(pop_pct * dop, 0.01, direction='down')
                except:
                    raise Exception('dop is ',dop,'pop_pct is ',pop_pct)
            else:
                target_buy_price = None
            potential_buy = False
            if target_buy_price is not None:
                if i > tib:
                    try:
                        if max(dct[sym][i-tib:i,0]) >= target_buy_price and max(dct[sym][:i-tib,0]) <= target_buy_price:
                            potential_buy = True
                    except ValueError:
                        print('sym', sym, 'i', i, 'todays_data_file', todays_data_file, 'rank', rank, 'len(dct[sym][:,0])', len(dct[sym][:,0]), 'ValueError on max')
                elif i == tib:
                    if max(dct[sym][:i,0]) >= target_buy_price:
                        potential_buy = True
                if potential_buy:
                    #Possible candidate for buying
                    vol = calc_vol_to_try_for(buying_power_wanted, target_buy_price, dct[sym][:i,1], p, t=1, go_back=280)
                    buying_power_needed, commission = calc_buying_power_needed(vol, target_buy_price)
                    if buying_power_needed <= buying_power_available and vol > 0:
                        # 2 * commission preemptively
                        buying_power_available -= round(vol * (target_buy_price + 2.0 * commission), 3)
                        buy_today.append([sym, target_buy_price, buying_power_needed, max(dct[sym][:i,0]), dop, pop_pct, hop_pct, stop_pct, vol, method, i - 1])
    return buy_today, buying_power_available
