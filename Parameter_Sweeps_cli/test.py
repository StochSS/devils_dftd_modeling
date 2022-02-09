#!/usr/bin/env python3


import sys
import time
import pickle
sys.path.insert(0,'../../GillesPy2/')
import gillespy2
from gillespy2 import Model, Species, Parameter, Reaction, Event, \
                      EventTrigger, EventAssignment, export_StochSS, \
                      RateRule
from gillespy2 import TauHybridCSolver

from DevilsDFTD2StageInfection import DevilsDFTD2StageInfectionVaccinationCullingImmunity






def main():
    variables = {
        "vaccinated_infection_rate": 0.60, # % of "break-through" cases
        "vaccination_proportion": 0.80,  # % of devil population that get vaccinated
        "vaccine_frequency": 2, #times per year we drop bait
        "vaccine_start": 444, # number of months after current date (444==Jan 2022)
        "vacc_program_length": 0, # number of years (10)
        "cull_start": 444,  # could be at 444 (current)
        "cull_program_length": 0, # number of years (10)
        "cull_rate_infected": 0.0,
        "cull_rate_diseased": 0.5,
        "immunity_growth_rate": 0.0075,
        "immunity_max_level": 70,
        "immunity_start": 0, # set to ~400 to turn on
    }
    
    tic=time.time()
    model = DevilsDFTD2StageInfectionVaccinationCullingImmunity(values=variables)
    solver = TauHybridCSolver(model=model, variable=True)
    results = []
    ext_count = 0
    erd_count = 0
    for _ in range(10):
        result = model.run(solver=solver, number_of_trajectories=1, variables=variables)
        # check to be sure the infectin took hold
        while result['Infected'][300] <= 0:
            result = model.run(solver=solver, number_of_trajectories=1, variables=variables)
        print(".",end='')
        sys.stdout.flush()
        results.append(result)
        Dftd = result['Infected'] + result['Exposed'] + result['Diseased']
        if min(Dftd[400:]) == 0.0:
            erd_count += 1
        Devils = Dftd + result['Juvenile'] + result['Susceptible'] + result['Vaccinated']
        if min(Devils[400:]) == 0.0:
            ext_count += 1

    with open("test_results.p","wb") as fd:
        pickle.dump(results,fd)
    print(f"done in {time.time()-tic}s")
    print(f"DFTD elimination: {erd_count}%")
    print(f"Devil extinction: {ext_count}%")


if __name__=="__main__":
    main()

