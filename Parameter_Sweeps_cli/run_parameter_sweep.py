#!/usr/bin/env python3

from collections import OrderedDict
from os.path import exists

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



def generate_parameter_space():
    params = OrderedDict()
    params["immunity_start"]= [384, 444, 504]
    params["immunity_growth_rate"]= [0.0075, 0.01, 0.0125]
    params["immunity_max_level"]= [50, 75, 100]
    params["vaccinated_infection_rate"]= [0.1, 0.2, 0.4, 0.6]
    params["vaccination_proportion"]= [0.6, 0.8, 1.0]
    params["vacc_program_length"]= [3, 5, 10, 15, 20]
    params["vaccine_frequency"]= [2, 4, 6]
    params["cull_rate_diseased"]= [0.25, 0.5, 0.75]
    params["cull_program_length"]= [3, 5, 10, 15, 20]
    return list(params.items())

def get_variables(ps_vec):
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
    space = generate_parameter_space()
    ret = variables.copy()
    for n,p in enumerate(ps_vec):
        #print(f"{space[n][0]}={space[n][1][p]}")
        ret[space[n][0]] = space[n][1][p]
    
    return ret
        

def ittr_parameter_space():
    #cnt = 0
    space = generate_parameter_space()
    vec = [0]*len(space)
    while True:
        yield vec
        #cnt+=1
        #if cnt>5: return False
        #i=0
        keep_updating=True
        while keep_updating:
            vec[i] += 1
            if vec[i]>=len(space[i][1]):
                vec[i]=0
                i+=1
                if i >= len(space):
                    return False
            else:
                keep_updating=False


def run_parameter_point(ps_vec_str):
    ps_vec = [ int(x) for x in ps_vec_str.split(",")]
    #print(f"run_parameter_point(ps_vec={ps_vec})")
    #return
    filename = f"dftd_ps-" + ",".join([str(x) for x in ps_vec])
    variables =  get_variables(ps_vec)
    if exists(filename + ".vars"):
        print(f"found {filename}.vars, skipping")
    with open(filename + ".vars", "wb") as fd:
       pickle.dump(variables,fd)

    return
    tic=time.time()
    model = DevilsDFTD2StageInfectionVaccinationCullingImmunity(values=variables)
    solver = TauHybridCSolver(model=model, variable=True)
    results = []
    ext_count = 0
    erd_count = 0
    for _ in range(100):
        result = model.run(solver=solver, number_of_trajectories=1, variables=variables)
        # check to be sure the infectin took hold
        inflc=0
        while result['Infected'][300] <= 0:
            inflc+=1
            if inflc>20: raise Exception("infinite loop: infection does not take hold")
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

    with open(filename+".elimination","w") as fd:
        fd.write(str(erd_count))
    with open(filename+".results","wb") as fd:
        pickle.dump(results,fd)
    #print(f"done in {time.time()-tic}s")
    #print(f"DFTD elimination: {erd_count}%")
    #print(f"Devil extinction: {ext_count}%")






def main():
    from multiprocessing import Pool
    all_space = []
    for ps_vec in ittr_parameter_space():
        all_space.append(",".join([str(x) for x in ps_vec]))
    with Pool(5) as p:
        p.map(run_parameter_point, all_space )


if __name__=="__main__":
    main()

