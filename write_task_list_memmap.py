import numpy as np
import sys
sys.path.append('/export/home/math/trose/python_utils')
import file_utils

def main():
    input_fname = 'method_3_params_8.csv'
    output_fname = 'method_3_task_list.dat'
    data_dir = '/export/home/math/trose/data/stocks/tasks/npz'

    num_data_files = len(file_utils.glob(file_utils.os.path.join(data_dir, '*.npz')))
    #num_data_files = 2
    
    csv_data = file_utils.read_csv(input_fname, map_type='float', dtype='float32')
    n = csv_data.shape[0]
    csv_data_0s = np.hstack((csv_data, np.zeros(n * 2, dtype='float32').reshape(n, 2)))
    for f in range(num_data_files):
        params_for_f = np.hstack((np.full((n, 1), f, dtype='float32'), csv_data_0s))
        if f == 0:
            all_tasks = params_for_f
        else:
            all_tasks = np.concatenate((all_tasks, params_for_f))

    fp = np.memmap(output_fname, dtype='float32', mode='w+', shape=all_tasks.shape)
    fp[:] = all_tasks[:]
    #print(fp)
main()
