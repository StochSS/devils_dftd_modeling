#!/usr/bin/env python3

import sys
variables={}
sweeps={}
try:
    for n,v in enumerate(sys.argv[1:]):
        print(f"{n}: {v}")
        if v.startswith('SWEEP_'):
            (vn,vv) = v.split('=')
            sn = vn.replace('SWEEP_','')
        else:
            (vn,vv) = v.split('=')
            variables[vn]=vv
except Exception as e:
    print(e)

print(variables)
