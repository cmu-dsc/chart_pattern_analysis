# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:17:58 2018

@author: timcrose
"""

import sys, os
import numpy as np
import random
from copy import deepcopy
sys.path.append('/export/home/math/trose/python_utils')

from simulation_approx import see_if_buy_executes, see_if_sell_executes, close_open_positions
import time_utils, file_utils, list_utils, math_utils
import real_time_find_matching_charts_old as fmc

start_time = time_utils.gtime()

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def remove_stocks_below_buy_threshold(tot_dct_full, pop_pct):
    return {sym:tot_dct_full[sym] for sym in tot_dct_full if max(tot_dct_full[sym][:,0]) >= round(tot_dct_full[sym][0,0] * pop_pct, 3)}

def main(todays_data_file, yesterdays_data_file, params, tot_dct_full):
    #params has form [param1, param2, param3, ..., method_number]
    max_buying_power_available = 1000.0
    buying_power_available = deepcopy(max_buying_power_available)
    
    todays_data_basename = os.path.basename(todays_data_file)
    todays_date = todays_data_basename[:todays_data_basename.find('.')]
    mon, day, yr = list(map(int, todays_date.split('_')))

    try:
        x = black_list
    except NameError:
        black_list = []
    #black_list has form [[sym, date_added]]
    for row in black_list:
        num_seconds_in_31_days = 31.0 * 24.0 * 3600.0
        #if can be removed from black_list
        if time_utils.get_greg_from_mdYHMS(mon, day, yr,  9, 0, 0) - row[1] > num_seconds_in_31_days:
            del black_list[black_list.index(row)]
            
    black_list_syms = [row[0] for row in black_list]

    if yesterdays_data_file != todays_data_file:
        tot_dct_full = np.load(todays_data_file)
    #tot_dct = remove_stocks_below_buy_threshold(tot_dct_full, params[0])
    if len(tot_dct) == 0:
        return 0.0, tot_dct_full
    end_of_day_greg = time_utils.get_greg_from_mdYHMS(mon, day, yr, 15, 45, 0)
    current_time = time_utils.get_greg_from_mdYHMS(mon, day, yr, 9, 30, 0)
    buy_today = []
    bought_today = []
    sold_today = []
    i = 1
    tib = 7
    while current_time < end_of_day_greg:
        current_time += 1
        i += 1
        #tib is the number of seconds that is pretty safe as to how long it takes to submit a trade thru IBgateway.
        if i % tib == 0:
            buy_today, buying_power_available = fmc.find_matching_charts(tot_dct, params, buy_today, max_buying_power_available, buying_power_available, tib, i, todays_data_file, rank)
        bought_today = see_if_buy_executes(tot_dct, buy_today, bought_today, tib - 1, i, h=2.0)
        sold_today, buying_power_available = see_if_sell_executes(tot_dct, sold_today, bought_today, buying_power_available, i, h=2.0)
        #delete syms that won't be used anymore today
        for item in sold_today:
            sold_sym = item[0]
            if sold_sym in tot_dct:
                del tot_dct[sold_sym]
    sold_today = close_open_positions(tot_dct, sold_today, bought_today, i)
    if len(sold_today) > 0:
        file_utils.write_rows_to_csv('results_' + str(rank) + '.csv', sold_today, mode='ab')
        file_utils.write_rows_to_csv('results_' + str(rank) + '.csv', [['###']], mode='ab')
        result = round(sum([item[-1] for item in sold_today]), 3)
    else:
        result = 0.0
    return result, tot_dct_full

def get_init_tasks(task_list_fname, num_data_files, num_params, pct_of_tasks_to_get, num_columns):
    float32_size = 4
    num_tot_tasks = num_data_files * num_params
    num_tasks_per_partition = num_tot_tasks / size
    #Initially start each task off with 10% of its equal share (to help less idle time)
    num_tasks_to_get = int(num_tasks_per_partition / pct_of_tasks_to_get)
    if num_tasks_to_get == 0:
        num_tasks_to_get = 1
    initial_idx = num_tasks_per_partition * rank
    offset = initial_idx * num_columns * float32_size
    shape = (num_tasks_to_get, num_columns)
    fp = np.memmap(task_list_fname, dtype='float32', mode='r+', offset=offset, shape=shape)
    print('fp_im', fp[:])
    #Change to ones to indicate running
    fp[:, 5] = 1
    print('fp_init', fp[:])
    return fp, num_tot_tasks, num_tasks_per_partition, num_tasks_to_get, initial_idx

def get_new_tasks(task_list_fname, start_row, num_tasks_to_get, num_columns, num_tot_tasks, initial_idx):
    float32_size = 4
    while True:
        if start_row >= num_tot_tasks:
            start_row = 0
        offset = start_row * num_columns * float32_size
        potential_end_pos = start_row + num_tasks_to_get
        if potential_end_pos >= num_tot_tasks:
            shape = (num_tot_tasks - start_row, num_columns)
        else:
            shape = (num_tasks_to_get, num_columns)
        fp = np.memmap(task_list_fname, dtype='float32', mode='r+', offset=offset, shape=shape)
        non_running_task_idxs = np.where(fp[:,5] == 0)[0]
        if non_running_task_idxs.shape[0] == 0:
            if start_row <= initial_idx and start_row + num_tasks_to_get > initial_idx:
                return None, None, True
            start_row += num_tasks_to_get
            continue
        #change to 1's to indicate running status
        fp[non_running_task_idxs, 5] = 1
        return fp, non_running_task_idxs, False
        #return fp[non_running_task_idxs,:], False

def run_main():
    wall_time = 10400.0
    time_limit = False
    data_dir = '/export/home/math/trose/data/stocks/tasks/npz'
    task_list_prefix = 'method_3_task_list'
    task_list_fname = task_list_prefix + '.dat'
    num_params = 8
    num_columns = 7
    #data_file_idx, pop_pct, hop_pct, stop_pct, num_trades_at_a_time, status, result
    pct_of_tasks_to_get = 10

    todays_data_files_list = file_utils.glob(file_utils.os.path.join(data_dir, '*.npz')) #[:2] ############### [:2] for DEBUGGING ONLY
    num_data_files = len(todays_data_files_list)

    start_time_get_fp = time_utils.gtime()
    fp, num_tot_tasks, num_tasks_per_partition, num_tasks_to_get, initial_idx = get_init_tasks(task_list_fname, num_data_files, num_params, pct_of_tasks_to_get, num_columns)
    print('fp', fp[:])

    non_running_task_idxs = [i for i in range(fp.shape[0])]
    yesterdays_data_file = ' '
    tot_dct = None
    times_per_day = []
    times_getting_task = []
    start_row = deepcopy(initial_idx)
    all_finished = False
    while not all_finished:
        t = 0
        while t < len(non_running_task_idxs): #fp.shape[0]:
            if time_utils.gtime() - start_time > wall_time - 65.0 and time_limit:
                #Change back to the 'hasn't run yet' state
                fp[non_running_task_idxs[t], 5] = 0
                break
            todays_data_file = todays_data_files_list[int(fp[non_running_task_idxs[t], 0])]
            params = fp[non_running_task_idxs[t], 1:5]
            start_time_for_day = time_utils.gtime()
            result, tot_dct = main(todays_data_file, yesterdays_data_file, params, tot_dct)
            fp[non_running_task_idxs[t], 6] = result
            #set to status to 2 (done)
            fp[non_running_task_idxs[t], 5] = 2
            yesterdays_data_file = todays_data_file
            times_per_day.append(time_utils.gtime() - start_time_for_day)
            t += 1
        start_row += num_tasks_to_get
        start_time_get_fp = time_utils.gtime()
        fp, non_running_task_idxs, all_finished = get_new_tasks(task_list_fname, start_row, num_tasks_to_get, num_columns, num_tot_tasks, initial_idx)
        
        times_getting_task.append(time_utils.gtime() - start_time_get_fp)
    #file_utils.write_row_to_csv('timings_' + str(rank) + '.csv', times_per_day)
    #file_utils.write_row_to_csv('timings_getting_task_' + str(rank) + '.csv', times_getting_task)
run_main()
file_utils.output_from_rank(message_args=('rank is done'), rank=rank)
