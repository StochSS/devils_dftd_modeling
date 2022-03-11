cd 'devils_dftd_modeling/Parameter_Sweeps'
screen                           # or "screen -r" if you have already started screen and want to re-attach
python3 -m venv dask_venv        # 1st time only
source dask_venv/bin/activate    # each time you start screen (not if you do "screen -r")
pip install gillespy2            # 1st time only
pip install dask[distributed]    # 1st time only

python

from dask.distributed import LocalCluster

cluster = LocalCluster(
host='james.cs.unca.edu',
scheduler_port=12345,
dashboard_address=None,
processes=True,
n_workers=55, 
threads_per_worker=1
)

cluster.adapt(minimum=1, maximum=55)  # Allows the cluster to auto scale to 55 when tasks are computed

# Ctl-A Ctl-D  to "detach" and keep it running
# Ctl-D to end python process/dask

