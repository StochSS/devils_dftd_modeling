
"""
GillesPy2 is a modeling toolkit for biochemical simulation.
Copyright (C) 2019-2021 GillesPy2 developers.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import os
import nbformat
import multiprocessing
from nbconvert.preprocessors import ExecutePreprocessor

root = '.'
notebooks = {}
errors = {}
jobs = []
ep = ExecutePreprocessor(timeout=600, kernel_name='python3', allow_errors=True)

def run(f, nb):
	try:
		ep.preprocess(f, {'metadata': {'path': root}})
		with open(nb, 'w', encoding='utf-8') as wf:
			nbformat.write(f, wf)
	except Exception as err:
		errors[nb] = err

def run_notebooks(run_dir):
	for root, dirs, files in os.walk(run_dir):
		for file in files:
			if file in ["Devils DFTD 2-Stage Infection with Immunity and Culling.ipynb",
					"Devils DFTD 2-Stage Infection with Immunity and Vaccination.ipynb",
					"Devils DFTD 2-Stage Infection with Vaccination and Culling.ipynb",
					"Devils DFTD 2-Stage Infection with Immunity, Vaccination, and Culling.ipynb"]:
				with open(os.path.join(root, file)) as f:
					print('Reading {}...'.format(file))
					notebooks[file] = nbformat.read(f, as_version=nbformat.NO_CONVERT)
	for nb, f in notebooks.items():
		p = multiprocessing.Process(target=run, name=nb, args=(f, nb,))
		jobs.append(p)

	for j in jobs:
		print(f'starting {j}')
		j.start()
	for j in jobs:
		print(f'joining {j}')
		j.join()






if __name__=='__main__':
    run_notebooks('.')
