#!/usr/bin/env python3

from collections import OrderedDict
from os.path import exists
import os

import sys
import time
import pickle



from DevilsDFTD2StageInfection import DevilsPreVaccination

type_variables = {
    "juvenile_concentration": float,
    "birth_rate": float,
    "maturity_rate": float,
    "death_rate_juvenile": float,
    "death_rate_susceptible": float,
    "death_rate_over_population": float,
    "infection_rate_infected": float,
    "infection_rate_diseased": float,
    "incubation": float,
    "progression": float,
    "death_rate_infected": float,
    "death_rate_diseased": float,
    "DFTD_start": float,
}
def get_variables():
    ''' take the string from argv and set variables'''
    variables={}
    sweeps={}
    try:
        for n,v in enumerate(sys.argv[1:]):
            #print(f"{n}: {v}")
            if v.startswith('SWEEP_'):
                (vn,vv) = v.split('=')
                sn = vn.replace('SWEEP_','')
                sv = [type_variables[sn](x) for x in vv.split(",")[0:-1]]
                sweeps[sn]=sv
            else:
                (vn,vv) = v.split('=')
                variables[vn]=vv
    except Exception as e:
        print(e)

    print(f"variables = {variables}") 
    print(f"sweeps = {sweeps}") 
    return (variables,sweeps)

#def make_filename(variables):
#    ret = os.path.dirname(__file__)+'/ps-'
#    for k in sorted(variables.keys()):
#        ret+=f"{variables[k]},"
#    return ret
def make_filename(variables):
    cmdpath="/home/brian/research/devils_dftd_modeling/Parameter_Sweeps_ssh/"
    ret = cmdpath+'/ps-'
    for k in sorted(variables.keys()):
        v = type_variables[k](variables[k])
        if type_variables[k]==int:
            x = str(v)
        else:
            x = f"{v:.4e}"
        ret+=f"{x},"
    return ret


def run_parameter_point(variables):
    filename = make_filename(variables)
    if exists(filename + ".p"):
        print(f"Found {filename}.p, skipping")
    print(f"Creating {filename}.p")

    model = DevilsPreVaccination(values=variables)
    results = model.run()
    (m,s) = model.calculate_distance(results)

    with open(filename+".m","w") as fd:
        fd.write(str(m)+','+str(s))
    with open(filename+".p","wb") as fd:
        pickle.dump(results,fd)


def generate_all_space():
    (variables,sweeps) = get_variables()
    # first add the base point
    all_space =[  variables.copy() ]
    # then go through each sweep and add the points
    for sn,svs in sweeps.items():
        for sv in svs:
            if variables[sn] == sv: continue # don't re-add base point
            v = variables.copy()
            v[sn] = sv
            all_space.append(v)
    return all_space

def main():
    from multiprocessing import Pool
    all_space = generate_all_space()
    with Pool(60) as p:
        p.map(run_parameter_point, all_space )


if __name__=="__main__":
    main()


