source /export/home/math/trose/anaconda2/envs/idp/bin/activate /export/home/math/trose/anaconda2/envs/idp
export I_MPI_SHM_LMT=shm
rm -f nohup.out results*.csv output_from_world_rank_* *lockfile* method_3_task_list_* timings_0.csv timings_1.csv *.pyc
rm -rf __pycache__
python write_task_list_memmap.py
time nohup mpirun -n 48 python real_time_trading_master.py &
