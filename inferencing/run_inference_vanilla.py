#!/usr/bin/env python3

import numpy as np
import sys
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(1, '/home/smatthe2/sciope')
sys.path.insert(1, '/home/smatthe2/GillesPy2')
import sciope
import gillespy2
from gillespy2 import Model, Species, Parameter, Reaction, Event, \
                      EventTrigger, EventAssignment
from gillespy2 import ODECSolver, ODESolver, SSACSolver, TauHybridCSolver
from tsfresh.feature_extraction.settings import MinimalFCParameters
from sciope.utilities.priors import uniform_prior
from sciope.utilities.summarystats import auto_tsfresh
from sciope.utilities.distancefunctions import naive_squared, euclidean, manhattan
from sciope.inference.abc_inference import ABC
from sklearn.metrics import mean_absolute_error
import dask
from dask.distributed import Client


variables = {'juvenile_concentration': 0.49534348836011316, 
			 'birth_rate': 0.055, 
			 'maturity_rate': 0.04, 
			 'infection_rate_infected': 1e-05, 
			 'infection_rate_diseased': 3.84e-05, 
			 'incubation': 10.25, 
			 'progression': 10.74, 
			 'death_rate_juvenile': 0.007, 
			 'death_rate_susceptible': 0.02335, 
			 'death_rate_over_population': 2.3e-07, 
			 'death_rate_infected': 0.022609, 
			 'death_rate_diseased': .29017, 
			 'DFTD_start': 100}

variables_orig = variables.copy()
total_time = np.arange(0, 421, 1)


pop_data = pd.read_csv('month_data/Devils_Dataset__Population_1985-2020.csv')

devil_pop = np.array(pop_data['Population'].iloc[:])
init_devils_pop = devil_pop[0]

obs = np.vstack([devil_pop]).reshape(1, 1, -1)
# print(obs)

class Devils2Stage(Model):
	def __init__(self, values=variables, events=None):
		Model.__init__(self, name="Devils DFTD 2-Stage Infection")
		self.volume = 1

		# Parameters
		birth_rate = Parameter(name="birth_rate", expression=values['birth_rate'])
		death_rate_juvenile = Parameter(name="death_rate_juvenile", expression=values['death_rate_juvenile'])
		maturity_rate = Parameter(name="maturity_rate", expression=values['maturity_rate'])
		death_rate_susceptible = Parameter(
			name="death_rate_susceptible", expression=values['death_rate_susceptible']
		)
		death_rate_over_population = Parameter(
			name="death_rate_over_population", expression=values['death_rate_over_population']
		)
		infection_rate_infected = Parameter(
			name="infection_rate_infected", expression=values['infection_rate_infected']
		)
		infection_rate_diseased = Parameter(
			name="infection_rate_diseased", expression=values['infection_rate_diseased']
		)
		incubation = Parameter(name="incubation", expression=values['incubation'])
		juvenile_concentration = Parameter(name="juvenile_concentration", expression=values['juvenile_concentration'])

		death_rate_infected = Parameter(name="death_rate_infected", expression=values['death_rate_infected'])
		progression = Parameter(name="progression", expression=values['progression'])
		death_rate_diseased = Parameter(name="death_rate_diseased", expression=values['death_rate_diseased'])
		DFTD_start = Parameter(name="DFTD_start", expression=values['DFTD_start'])
		
		self.add_parameter([birth_rate, death_rate_juvenile, maturity_rate, death_rate_susceptible, juvenile_concentration,
							death_rate_over_population, infection_rate_infected, infection_rate_diseased,
							incubation, death_rate_infected, progression, death_rate_diseased, DFTD_start])

		# Variables (initial values adjusted to observed data)
		initial_devil_population  = int(devil_pop[0])
		
		Juvenile = Species(
			name="Juvenile", mode="discrete",
			initial_value=round(initial_devil_population * values['juvenile_concentration'])
		)
		Susceptible = Species(
			name="Susceptible", mode="discrete",
			initial_value=round(initial_devil_population * (1 - values['juvenile_concentration']))
		)
		Exposed = Species(name="Exposed", initial_value=0, mode="discrete")
		Infected = Species(name="Infected", initial_value=0, mode="discrete")
		Diseased = Species(name="Diseased", initial_value=0, mode="discrete")
		Devils = Species(name="Devils", initial_value=initial_devil_population, mode="discrete")
		self.add_species([Juvenile, Susceptible, Exposed, Infected, Diseased, Devils])

				
		# Reactions
		Birth = Reaction(name="Birth",
			reactants={}, products={'Juvenile': 1, 'Devils': 1},
			propensity_function="birth_rate * (Susceptible + Exposed + Infected)"
		)
		Mature = Reaction(name="Mature",
			reactants={'Juvenile': 1}, products={'Susceptible': 1},
			propensity_function="Juvenile * maturity_rate"
		)
		Death_Diseased = Reaction(name="Death_Diseased",
			reactants={'Diseased': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_diseased * Diseased"
		)
		Death_Diseased2 = Reaction(name="Death_Diseased2",
			reactants={'Diseased': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_over_population * Diseased * (Devils - 1)"
		)
		Death_Exposed = Reaction(name="Death_Exposed",
			reactants={'Devils': 1, 'Exposed': 1}, products={},
			propensity_function="death_rate_susceptible * Exposed"
		)
		Death_Exposed2 = Reaction(name="Death_Exposed2",
			reactants={'Devils': 1, 'Exposed': 1}, products={},
			propensity_function="death_rate_over_population * Exposed * (Devils - 1)"
		)
		Death_Infected = Reaction(name="Death_Infected",
			reactants={'Infected': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_infected * Infected"
		)
		Death_Infected2 = Reaction(name="Death_Infected2",
			reactants={'Infected': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_over_population * Infected * (Devils-1)"
		)
		Death_Juvenile = Reaction(name="Death_Juvenile",
			reactants={'Juvenile': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_juvenile * Juvenile"
		)
		Death_Juvenile2 = Reaction(name="Death_Juvenile2",
			reactants={'Juvenile': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_over_population * Juvenile * (Devils-1)"
		)
		Death_Susceptible = Reaction(name="Death_Susceptible",
			reactants={'Susceptible': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_susceptible * Susceptible"
		)
		Death_Susceptible2 = Reaction(name="Death_Susceptible2",
			reactants={'Susceptible': 1, 'Devils': 1}, products={},
			propensity_function="death_rate_over_population * Susceptible * (Devils-1)"
		)
		DFTD_Stage1 = Reaction(name="DFTD_Stage1",
			reactants={'Exposed': 1}, products={'Infected': 1},
			propensity_function="Exposed / incubation"
		)
		DFTD_Stage2 = Reaction(name="DFTD_Stage2",
			reactants={'Infected': 1}, products={'Diseased': 1},
			propensity_function="Infected / progression"
		)
		TransmissionD = Reaction(name="TransmissionD",
			reactants={'Susceptible': 1, 'Diseased': 1}, products={'Exposed': 1, 'Diseased': 1},
			propensity_function="infection_rate_diseased * Susceptible * Diseased"
		)
		TransmissionI = Reaction(name="TransmissionI",
			reactants={'Susceptible': 1, 'Infected': 1}, products={'Exposed': 1, 'Infected': 1},
			propensity_function="infection_rate_infected * Susceptible * Infected"
		)
		self.add_reaction([
			Birth, Mature, Death_Diseased, Death_Diseased2, Death_Exposed, Death_Exposed2, Death_Infected,
			Death_Infected2, Death_Juvenile, Death_Juvenile2, Death_Susceptible, Death_Susceptible2,
			DFTD_Stage1, DFTD_Stage2, TransmissionD, TransmissionI
		])

		# Events
		
		et = EventTrigger(expression='t>=DFTD_start')
		ea1 = EventAssignment(variable=Susceptible, expression='Susceptible-1')
		ea2 = EventAssignment(variable=Infected, expression='1')
		introduce_dftd = Event(name='introduce_dftd', trigger=et, assignments=[ea1, ea2])
		self.add_event(introduce_dftd)
		if events is not None:
			self.add_event(events)

		# Timespan
		self.timespan(np.arange(0, 421, 1)) # month data tspan
		


model = Devils2Stage()
solver = TauHybridCSolver(model=model, variable=True)

def configure_simulation():
	solver = TauHybridCSolver(model=model, variable=True)
	kwargs = {
		"solver":solver
	}
	return kwargs

kwargs = configure_simulation()

def main():
	'''
	DEFINE PRIORS
	'''
	default_param = np.array(list(model.listOfParameters.items()))[:, 1]

	parameter_names = []
	bound = []
	mat_ind = 100
	for i,exp in enumerate(default_param):
		if exp.name=='maturity_rate':
			mat_ind = i
		bound.append(float(exp.expression))
		parameter_names.append(exp.name)

	# Set the bounds
	bound = np.array(bound)
	min_stand = bound * .75
	max_stand = bound * 1.25
	min_stand[mat_ind] = bound[mat_ind]
	max_stand[mat_ind] = bound[mat_ind]

	dmin = np.log(min_stand)
	dmax = np.log(max_stand)

	print(dmin)
	print(dmax)
	# Here we use uniform prior
	uni_prior = uniform_prior.UniformPrior(dmin, dmax)

	'''
	DEFINE SIMULATOR
	'''
	def set_model_parameters(params):
		# params - array, need to have the same order as model.listOfParameters
		variables = dict(zip(parameter_names, params))
		return variables

	# Here we use the GillesPy2 Solver
	def simulator(params, model):
		print('testing params:\n', params)
		params = np.exp(params)
		variables = set_model_parameters(params)
		print(variables)
	#     model_update = set_model_parameters(params, model)
		res = model.run(solver=solver)
		devils = res['Devils']
		res = res.to_array()
		tot_res = np.asarray([x.T for x in res]) # reshape to (N, S, T)
		# should not contain timepoints
		tot_res = tot_res[:, 1:, :]
		
	#     infected = res['I']

		return np.vstack([devils]).reshape(1, 1, -1)
	#     return tot_res

	# Wrapper, simulator function to abc should should only take one argument (the parameter point)
	def simulator2(x):
		return simulator(x, model=model)

	# Function to generate summary statistics
	summ_func = auto_tsfresh.SummariesTSFRESH()

	# Distance
	ed = euclidean.EuclideanDistance()

	c = Client('james.cs.unca.edu:12345')
	print(c)

	abc = ABC(obs,
							  sim=simulator2,
							  prior_function=uni_prior,
							  summaries_function=summ_func.compute,
							  distance_function=ed
							 )

	abc.compute_fixed_mean(chunk_size=2)
	res = abc.infer(num_samples=100, batch_size=10, chunk_size=2)

	with open('abc_res_vanilla.p', 'wb') as res_obj_file:
		pickle.dump(res, res_obj_file)

	mae_inference = mean_absolute_error(bound, abc.results['inferred_parameters'])
	posterior = np.array(res[0]['accepted_samples'])


	fig, ax = plt.subplots(posterior.shape[1], posterior.shape[1])

	for i in range(posterior.shape[1]):
		for j in range(posterior.shape[1]):
			if i > j:
				ax[i,j].axis('off')
			else:
				if i == j:
					ax[i,j].hist(np.exp(posterior[:,i]), bins = 'auto')
					ax[i,j].axvline(np.median(np.exp(posterior[:,i])), color = 'C1')
					ax[i,j].set_xlim(np.exp(dmin[i]), np.exp(dmax[i]))
				else:
					ax[i,j].scatter(np.exp(posterior[:,j]), np.exp(posterior[:,i]))
					ax[i,j].set_ylim(np.exp(dmin[i]), np.exp(dmax[i]))
					ax[i,j].set_xlim(np.exp(dmin[j]), np.exp(dmax[j]))
		print('{}: {}'.format(i, parameter_names[i]))
		ax[i,0].set_ylabel(parameter_names[i])
		ax[0,i].set_title(parameter_names[i])
	fig.set_size_inches(48,48)
	plt.savefig('devil-param-correlation-run2.png')
	with open('fig-obj-run-vanilla.p', 'wb') as fig_obj_file:
		pickle.dump(fig, fig_obj_file)
	with open('posterior-data-run-vanilla.p', 'wb') as posterior_data_file:
		pickle.dump(posterior, posterior_data_file)

	selected_vars = dict(zip(parameter_names, np.exp(posterior[best_ind])))

	test = model.run(solver=solver, variables=selected_vars)
	with open('selected_vars_vanilla.p', 'wb') as selected_vars_file:
		pickle.dump(selected_vars, selected_vars_file)

if __name__=='__main__':
	main()
