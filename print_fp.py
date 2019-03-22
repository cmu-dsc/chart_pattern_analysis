import numpy as np
import sys
sys.path.append('/export/home/math/trose/python_utils')
import file_utils

def get_pct_of_tasks_done(fp, num_rows):
    num_finished_tasks = float(len(np.where(fp[:,5] == 2)[0]))
    return num_finished_tasks / float(num_rows)

def main():
    input_fname = 'method_3_task_list.dat'
    num_columns = 7
    num_params = 8
    data_dir = '/export/home/math/trose/data/stocks/tasks/npz'
    #num_data_files = 2
    todays_data_files_list = file_utils.glob(file_utils.os.path.join(data_dir, '*.npz')) #[:2] ############### [:2] for DEBUGGING ONLY
    num_data_files = len(todays_data_files_list)

    num_rows = num_params * num_data_files
    shape = (num_rows, num_columns)

    #np.set_printoptions(threshold=np.nan)
    fp = np.memmap(input_fname, dtype='float32', mode='r', shape=shape)
    print('fp.shape',fp.shape)
    #print(fp[:])
    print(fp[np.where(fp[:,3] == 0)])
    #pct_of_tasks_done = get_pct_of_tasks_done(fp, num_rows)
    #print(pct_of_tasks_done)
main()

