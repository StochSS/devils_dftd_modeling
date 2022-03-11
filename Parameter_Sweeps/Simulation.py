import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dask import delayed, compute

pop_data = pd.read_csv('../month_data/Devils_Dataset__Population_1985-2020.csv')
devil_pop = np.array(pop_data['Population'].iloc[:].values)

dates = []
year = 1985
while len(dates) < 1001:
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]:
        dates.append(f"{month} {year}")
    year += 1

class Simulation:
    def __init__(self, model, kwargs=None, variables=None):
        self.result = None
        self.variables = variables
        self.model = model
        self.kwargs = kwargs
        self.dftd_elimination = None
        self.devil_extinction = None
        
    def __compute_dftd_prob(self, result):
        Dftd = result['Infected'] + result['Exposed'] + result['Diseased']
        if min(Dftd[400:]) == 0.0:
            self.dftd_elimination += 1
        return Dftd
    
    def __compute_devil_prob(self, result, Dftd):
        Devils = Dftd + result['Juvenile'] + result['Susceptible']
        if "Vaccinated" in result:
            Devils += result['Vaccinated']
        if min(Devils[400:]) == 0.0:
            self.devil_extinction += 1
    
    def __load_dask_sims(self, sim_count):
        if self.kwargs is None:
            self.configure()
        prob_sims = []
        for _ in range(sim_count):
            sim_thread = delayed(self.model.run)(**self.kwargs)
            prob_sims.append(sim_thread)
        return prob_sims
    
    def output_dftd_devils_probs(self, print_probs=False):
        if print_probs:
            print(f"DFTD elimination: {self.dftd_elimination}%")
            print(f"Devil extinction: {self.devil_extinction}%")
            return
        return self.dftd_elimination, self.devil_extinction
    
    def configure(self, solver=None):
        self.kwargs = {
            "number_of_trajectories": 1
        }
        if solver is not None:
            self.kwargs['solver'] = solver
    
    @classmethod
    def load_state(cls, state):
        try:
            sim = Simulation(state.model, kwargs=state.kwargs, variables=state.variables)
        except:
            sim = Simulation(state.model, kwargs=state.kwagrs, variables=state.variables)
        sim.result = state.result
        sim.dftd_elimination = state.dftd_elimination
        sim.devil_extinction = state.devil_extinction
        return sim
    
    def plot(self, start=0, alpha=0.3, plot_observed=False, plot_immunity_level=True):
        carry_cap = int(max(devil_pop)*1.16)
        dftd_start = int(self.result.model.listOfParameters['DFTD_introduction'].value)
        
        spec_list = [self.result['Juvenile'], self.result['Susceptible'], self.result['Exposed'],
                     self.result['Infected'], self.result['Diseased']]
        if "Vaccinated" in self.result[0].data:
            spec_list.append(self.result['Vaccinated'])
        total_devils = np.add.reduce(spec_list)
        x = self.result['time'][start:]
        text_offset = (self.result['time'].size - start) / 601
        
        fig, ax1 = plt.subplots(figsize=[15, 8])
        interventions = []
        if "immunity_start" in self.result.model.listOfParameters:
            interventions.append("Immunity")
        if "Vaccinated" in self.result[0].data:
            interventions.append("Vaccination")
        if "culling_start" in self.result.model.listOfParameters:
            interventions.append("Culling")
        interventions = " + ".join(interventions)
        plt.title(f"Tasmanian Devil Population with DFTD: {interventions} Program", fontsize=18)
        ax1.set_xlabel(f"Time (months) since {dates[start]}", fontsize=14)
        ax1.set_ylabel("Population of Tasmanian Devils", fontsize=14)
        ax1.plot(x, total_devils[start:], color='blue', label='Total Devils')
        ax1.plot(x, self.result['Juvenile'][start:], color='purple', alpha=alpha, label='Juvenile')
        ax1.plot(x, self.result['Susceptible'][start:], color='green', alpha=alpha, label='Susceptible')
        ax1.plot(x, self.result['Exposed'][start:], color='magenta', alpha=alpha, label='Exposed')
        ax1.plot(x, self.result['Infected'][start:], color='red', alpha=alpha, label='Infected')
        ax1.plot(x, self.result['Diseased'][start:], color='brown', alpha=alpha, label='Diseased')
        
        if plot_observed:
            ax1.plot(range(len(devil_pop)), devil_pop, '--k', label='Observed')
        
        # DFTD Introduction
        if start <= dftd_start:
            ax1.plot([dftd_start, dftd_start], [-3000, carry_cap], '--k', alpha=0.3)
            ax1.text(dftd_start - 10 * text_offset, 45000, "DFTD Introduced",
                     rotation="vertical", color="black", fontsize=12)
            ax1.text(dftd_start + 3 * text_offset, 48000, dates[dftd_start],
                     rotation="vertical", color="black", fontsize=12)
        
        # Immunity
        if "immunity_start" in self.result.model.listOfParameters:
            if self.variables is not None and "immunity_start" in self.variables.keys():
                immunity_start = int(self.variables['immunity_start'])
            else: 
                immunity_start = int(self.result.model.listOfParameters['immunity_start'].value)
            
            if immunity_start > 0:
                ax1.plot([immunity_start, immunity_start], [-3000, carry_cap], '--k', alpha=0.3)
                ax1.text(immunity_start - 10 * text_offset, 28000, f"Start Immunity: {dates[immunity_start]}",
                         rotation="vertical", color="black", fontsize=12)

                if plot_immunity_level:
                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                    ax2.plot(self.result['immunity_level'], '--r', alpha=0.3, label="immunity")
                    ax2.set_ylim(0,100)
                    ax2.set_yticks(ax2.get_yticks())
                    ax2.set_ylabel("Immunity %", color="red")
                    ax2.tick_params(axis='y', labelcolor="red")

        # Vaccination
        if "Vaccinated" in self.result[0].data:
            if self.variables is not None and "vaccine_start" in self.variables.keys():
                vaccine_start = self.variables['vaccine_start']
            else:
                vaccine_start = int(self.result.model.listOfParameters['vaccine_start'].value)

            if self.variables is None or 'vacc_program_length' not in self.variables:
                vacc_program_length = self.result.model.listOfParameters['vacc_program_length'].value
                vaccine_end = vaccine_start + 12 * int(vacc_program_length)
            else:
                vaccine_end = vaccine_start + 12 * int(self.variables['vacc_program_length'])
            
            if vaccine_start < vaccine_end:
                ax1.plot(x, self.result['Vaccinated'][start:], color='cyan', alpha=alpha, label='Vaccinated')
                ax1.plot([vaccine_start, vaccine_start], [-3000, carry_cap - 3000], '--k', alpha=0.3)
                ax1.plot([vaccine_end, vaccine_end], [-3000, carry_cap - 3000], '--k', alpha=0.3)
                ax1.plot([vaccine_start, vaccine_end], [carry_cap - 3000, carry_cap - 3000], '--k', alpha=0.3)
                ax1.text(
                    vaccine_start, carry_cap - 2300, f"Vaccine: {dates[vaccine_start]} - {dates[vaccine_end]}",
                    color="black", fontsize=12
                )
            
        if "culling_start" in self.result.model.listOfParameters:
            if self.variables is not None and "culling_start" in self.variables.keys():
                culling_start = self.variables['culling_start']
            else:
                culling_start = int(self.result.model.listOfParameters['culling_start'].value)

            if self.variables is None or 'cull_program_length' not in self.variables:
                cull_program_length = self.result.model.listOfParameters['cull_program_length'].value
                culling_end = culling_start + 12 * int(cull_program_length)
            else:
                culling_end = culling_start + 12 * int(self.variables['cull_program_length'])
                
            if culling_start < culling_end:
                ax1.plot([culling_start, culling_start], [-3000, carry_cap - 8000], '--k', alpha=0.3)
                ax1.plot([culling_end, culling_end], [-3000, carry_cap - 8000], '--k', alpha=0.3)
                ax1.plot([culling_start, culling_end], [carry_cap - 8000, carry_cap - 8000], '--k', alpha=0.3)
                ax1.text(
                    culling_start, carry_cap - 7300, f"Culling: {dates[culling_start]} - {dates[culling_end]}",
                    color="black", fontsize=12
                )
        
        ax1.set_ylim(-3000, carry_cap)
        ax1.set_xlim(-5, 1005)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.tick_params(axis='y',labelsize=12, labelrotation=90)
        ax1.legend(loc='upper right', fontsize=12)
        fig.tight_layout()
    
    def run(self, return_results=False, use_existing_results=False, verbose=False):
        if self.result is not None and use_existing_results:
            return
        
        dask_sims = self.__load_dask_sims(100)
        dask_results = compute(*dask_sims)
        
        failed_attempts = 0
        self.dftd_elimination = 0
        self.devil_extinction = 0
        for (result, attempts) in dask_results:
            if verbose: print(".", end='')
            Dftd = self.__compute_dftd_prob(result)
            self.__compute_devil_prob(result, Dftd)
            failed_attempts += attempts
        
        if verbose: print(f"'\nFailed Attempts: {failed_attempts}")
        if return_results:
            return dask_results[0][0]
        self.result = dask_results[0][0]
        return self