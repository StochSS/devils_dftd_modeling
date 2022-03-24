import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), '../../GillesPy2')))

import time
import json
import pickle

from gillespy2 import TauHybridCSolver

from dask.distributed import Client
from dask import delayed

from Devils_DFTD_2_Stage_Infection import DevilsDFTD2StageInfection
from Simulation import Simulation, devil_pop
from ParameterSweep import ParameterSweep

c = Client("james.cs.unca.edu:12345")

# Devils DFTD 2-Stage Infection With Vaccination + Culling Interventions
def load_existing_state(state_path):
	if os.path.exists(state_path):
	    with open(state_path, "rb") as state_file:
	        state = pickle.load(state_file)
	        
	    model = state['model']
	    sim = Simulation.load_state(state['sim'])
	    job = ParameterSweep.load_state(state['job'], batch_size=150, statefile=state_path)
		return model, sim, job

    model = DevilsDFTD2StageInfection(devil_pop, interventions=["vaccination", "culling"])
    sim = Simulation(model=model)
    job = ParameterSweep(model=model, batch_size=150, statefile=state_path)
    return model, sim, job

def run_parameter_sweep__vacc_cull():
	state_path = "DevilsDFTD2StageInfectionWithVaccinationAndCullingState.p"
	model, sim, job = load_existing_state(state_path)

	sol = delayed(TauHybridCSolver)(model=model, variable=True, delete_directory=False)

	sim.configure(solver=sol)
	_ = sim.run(use_existing_results=True)

	sim.output_dftd_devils_probs(print_probs=True)

	params = [
	    {"parameter": "vaccinated_infection_rate", "range": [0.1, 0.2, 0.4, 0.6]},
	    {"parameter": "vaccination_proportion", "range": [0.6, 0.8, 1.0]},
	    {"parameter": "vacc_program_length", "range": [3, 5, 6, 7, 8, 9, 10, 11]},
	    {"parameter": "vaccine_frequency", "range": [2, 4, 6]},
	    {"parameter": "cull_rate_diseased", "range": [0.25, 0.5, 0.75]},
	    {"parameter": "cull_program_length", "range": [3, 5, 6, 7, 8, 9, 10, 11]}
	]
	total_param_points = 1
	for param in params:
	    total_param_points *= len(param['range'])
	print(f"Size of the Parameter Space: {total_param_points}")

	job.run(solver=sol, params=params)

	state = {"model": model, "sim":sim, "job":job}
	with open(state_path, "wb") as state_file:
	    pickle.dump(state, state_file)

if __name__ == "__main__":
	run_parameter_sweep__vacc_cull()
