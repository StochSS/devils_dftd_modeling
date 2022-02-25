cd 'devils_dftd_modeling/Parameter_Sweeps'
python3 -m venv dask_venv
source dask_venv/bin/activate
pip install gillespy2
pip install dask[distributed]
python

from dask.distributed import LocalCluster

cluster = LocalCluster(
host='james.cs.unca.edu',
scheduler_port=12345,
dashboard_address=None,
processes=True,
n_workers=45, 
threads_per_worker=1
)

#cluster.adapt(minimum=1, maximum=45)  # Allows the cluster to auto scale to 50 when tasks are computed

