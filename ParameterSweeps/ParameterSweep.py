import copy
import json
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output
import ipywidgets as widgets
from IPython.display import display

from dask import delayed, compute

from Simulation import Simulation

dates = []
year = 1985
while len(dates) < 1001:
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]:
        dates.append(f"{month} {year}")
    year += 1

with open("./units_labels.json", "r") as ul_file:
    units_labels = json.load(ul_file)

class ParameterSweep():
    def __init__(self, model, params=None):
        self.model = model
        self.params = params
        self.results = {}
        
        self.result_keys = []
        self.simulations = []
        self.load_data_job = []

    def __get_result_key(self, variables):
        elements = []
        for name, value in variables.items():
            elements.append(f"{name}:{value}")
        return ",".join(elements)

    def __load(self, solver, index, variables, verbose):
        if index < len(self.params):
            param = self.params[index]
            index += 1
            for val in param['range']:
                variables[param['parameter']] = val
                self.__load(solver=solver, index=index, variables=variables, verbose=verbose)
        else:
            result_key = self.__get_result_key(variables=variables)
            if result_key not in self.results:
                if verbose:
                    message = f'adding job: {result_key.replace(":", "=").replace(",", ", ")}'
                    print(message)
                tmp_sim = Simulation(model=self.model, variables=copy.deepcopy(variables))
                tmp_sim.configure(solver=solver)
                tmp_sim.kwargs['variables'] = variables.copy()
                sim_thread = delayed(tmp_sim.run)(verbose=verbose)
                self.simulations.append(sim_thread)
                self.result_keys.append(result_key)
                
    def __run(self):
        results = dict(zip(self.result_keys, compute(*self.simulations)))
        if self.results:
            unsorted_keys = list(self.results.keys())
            unsorted_keys.extend(list(results.keys()))
            keys = self.__sort_keys(unsorted_keys)
            new_results = {}
            for key in keys:
                if key in results:
                    new_results[key] = results[key]
                else:
                    new_results[key] = self.results[key]
            self.results = new_results
        else:
            self.results = results

    @classmethod
    def __sort_keys(cls, keys):
        sort_map = {f"vacc_program_length:{i}": f"vacc_program_length:0{i}" for i in range(1, 10)}
        sort_map.update({f"cull_program_length:{i}": f"cull_program_length:0{i}" for i in range(1, 10)})
        sort_map.update({f"immunity_max_level:{i}": f"immunity_max_level:0{i}" for i in range(50, 100, 5)})
        rsort_map = {}
        for key, val in sort_map.items():
            rsort_map[val] = key

        unsorted_keys = []
        for key in keys:
            sub_keys = []
            for sub_key in key.split(','):
                if sub_key in sort_map:
                    sub_key = sort_map[sub_key]
                sub_keys.append(sub_key)
            unsorted_keys.append(','.join(sub_keys))
        unsorted_keys.sort()

        sorted_keys = []
        for key in unsorted_keys:
            sub_keys = []
            for sub_key in key.split(','):
                if sub_key in rsort_map:
                    sub_key = rsort_map[sub_key]
                sub_keys.append(sub_key)
            sorted_keys.append(','.join(sub_keys))
        return sorted_keys
    
    def build_layout(self, ai_widgets):
        ai_widgets = list(ai_widgets.values())
        hbs = []
        for i in range(0, len(ai_widgets), 4):
            hb_list = [ai_widgets[i], ai_widgets[i+1]]
            if len(ai_widgets) >= i+3:
                hb_list.extend([ai_widgets[i+2], ai_widgets[i+3]])
            hbs.append(widgets.HBox(hb_list, layout=self.get_layout()))
        return widgets.VBox(hbs, layout=self.get_layout(vertical=True))
    
    def build_widgets(self):
        param_names = units_labels['w_labels']
        ai_widgets = {}
        for i, param in enumerate(self.params):
            fs = widgets.SelectionSlider(
                options=param['range'], value=param['range'][0], description=param_names[param['parameter']]
            )
            ai_widgets[f'fs{i}'] = fs
            cs = widgets.Checkbox(value=False, description='Fixed')
            ai_widgets[f'cs{i}'] = cs
        return ai_widgets
    
    def configure(self, **widget_args):
        sim_key = []
        for i in range(0, len(widget_args), 2):
            param_key = int(i/2)
            sim_key.append(f"{self.params[param_key]['parameter']}:{list(widget_args.values())[i]}")
        sim_key = ",".join(sim_key)
        
        self.results[sim_key].plot(plot_observed=self.plot_observed)
        
        params, fixed = self.display_details(widget_args)

        if len(params) < 1:
            print("At least 1 fixed parameters are required")
        elif len(params) > 2:
            print("There are too many fixed parameters")
        elif len(params) == 2:
            base_key = self.get_base_key(list(widget_args.values())[::2], params)
            dftd, devils = self.get_plot_data(params, base_key)
            self.display_plots(params, *dftd, *devils)
        else:
            labels = units_labels['labels']
            units = units_labels['units']
            param = params[0]
            self.plot_devil_dftd_extinction_over_param(
                res_sub_keys=fixed, key=param['parameter'], param_label=labels[param['parameter']],
                units=units[param['parameter']]
            )
    
    def display_details(self, args, verbose=False):
        params = []
        fixed = []
        values = list(args.values())
        for i in range(0, len(values), 2):
            index = int(i/2)
            if values[i + 1]:
                fixed.append(f"{self.params[index]['parameter']}: {values[i]}")
            else:
                params.append(self.params[index])
        if fixed and verbose:
            print(", ".join(fixed))
        return params, [param.replace(": ", ":") for param in fixed]
    
    def display_plots(self, params, dftd, dftd_cflip, devils, devils_cflip):
        labels = units_labels['labels']
        units = units_labels['units']
        x_units = units[params[0]['parameter']]
        if x_units:
            x_units = f" ({x_units})"
        y_units = units[params[1]['parameter']]
        if y_units:
            y_units = f" ({y_units})"
        x_label = f"{labels[params[0]['parameter']]}{x_units}"
        y_label = f"{labels[params[1]['parameter']]}{y_units}"
        dftd = np.flip(dftd, 0)
        devils = np.flip(devils, 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16, 6])

        im1 = ax1.imshow(dftd)
        ax1.set_xticks(np.arange(len(dftd[0])))
        ax1.set_xticklabels(labels=params[0]['range'])
        ax1.set_yticks(np.arange(len(dftd)))
        ax1.set_yticklabels(labels=np.flip(params[1]['range']))
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(y_label, fontsize=14)
        ax1.tick_params(axis="x", labelsize=12, labelrotation=90)
        ax1.tick_params(axis="y", labelsize=12)
        ax1.set_title('Probability of DFTD Elimination', fontsize=14)
        ax1.figure.colorbar(im1, ax=ax1)
        for i in range(len(dftd)):
            for j in range(len(dftd[0])):
                color = "black" if dftd[i, j] > dftd_cflip else "w"
                _ = ax1.text(j, i, f"{dftd[i, j]}%", ha="center", va="center", color=color, fontsize=12)

        im2 = ax2.imshow(devils)
        ax2.set_xticks(np.arange(len(devils[0])))
        ax2.set_xticklabels(labels=params[0]['range'])
        ax2.set_yticks(np.arange(len(devils)))
        ax2.set_yticklabels(labels=np.flip(params[1]['range']))
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel(y_label, fontsize=14)
        ax2.tick_params(axis="x", labelsize=12, labelrotation=90)
        ax2.tick_params(axis="y", labelsize=12)
        ax2.set_title('Probability of Devil Extinction', fontsize=14)
        ax2.figure.colorbar(im2, ax=ax2)
        for i in range(len(devils)):
            for j in range(len(devils[0])):
                color = "black" if devils[i, j] > devils_cflip else "w"
                _ = ax2.text(j, i, f"{devils[i, j]}%", ha="center", va="center", color=color, fontsize=12)
    
    def explore_results(self, plot_observed=False):
        self.plot_observed = plot_observed
        ai_widgets = self.build_widgets()
        ui = self.build_layout(ai_widgets)
        out = interactive_output(self.configure, ai_widgets)
        display(ui, out)
    
    def get_base_key(self, values, params):
        base_key = []
        for i, param in enumerate(self.params):
            if param in params:
                base_key.append("__param2__" if "__param1__" in base_key else "__param1__")
            else:
                base_key.append(f"{param['parameter']}:{values[i]}")
        return ",".join(base_key)
    
    def get_layout(self, vertical=False):
        kwargs = {
            "margin": '0px 10px 10px 0px',
            "padding": '5px 5px 5px 5px'
        }
        if vertical:
            kwargs['border'] = 'solid 1px red'
        return widgets.Layout(**kwargs)
    
    def get_plot_data(self, params, base_key):
        dftd = []
        dftd_lim = [100, 0]
        devils = []
        devils_lim = [100, 0]
        for value1 in params[1]['range']:
            _key = base_key.replace("__param2__", f"{params[1]['parameter']}:{value1}")
            inner_dftd = []
            inner_devils = []
            for value2 in params[0]['range']:
                key = _key.replace("__param1__", "{0}:{1}".format(params[0]['parameter'], value2))
                dftd_prob, devil_prob = self.results[key].output_dftd_devils_probs()
                inner_dftd.append(dftd_prob)
                inner_devils.append(devil_prob)
            if min(inner_dftd) < dftd_lim[0]:
                dftd_lim[0] = min(inner_dftd)
            if max(inner_dftd) > dftd_lim[1]:
                dftd_lim[1] = max(inner_dftd)
            if min(inner_devils) < devils_lim[0]:
                devils_lim[0] = min(inner_devils)
            if max(inner_devils) > devils_lim[1]:
                devils_lim[1] = max(inner_devils)
            dftd.append(inner_dftd)
            devils.append(inner_devils)
        dftd_cflip = dftd_lim[1] - ((dftd_lim[1] - dftd_lim[0]) * 0.3)
        devils_cflip = devils_lim[1] - ((devils_lim[1] - devils_lim[0]) * 0.3)
        return (np.array(dftd), dftd_cflip), (np.array(devils), devils_cflip)
    
    def get_devil_dftd_extinction_over_param(self, res_sub_keys, key=None, return_data=False, verbose=False):
        if len(self.params) < 2:
            keys = self.results.keys()
        elif (len(self.params) - len(res_sub_keys)) != 1:
            raise Exception(f"res_sub_keys[{len(self.params)}] must be set.")
        else:
            _keys = list(self.results.keys())
            for sub_key in res_sub_keys:
                keys = []
                for res_key in _keys:
                    if sub_key in res_key.split(","):
                        keys.append(res_key)
                _keys = keys
            
        pl_values = []
        pl_ext_rate = []
        pl_erd_rate = []
        for res_key in keys:
            pl_values.append(self.results[res_key].variables[key])
            dftd_prob, devil_prob = self.results[res_key].output_dftd_devils_probs()
            pl_ext_rate.append(devil_prob)
            pl_erd_rate.append(dftd_prob)
        return pl_values, pl_erd_rate, pl_ext_rate
        
    @classmethod
    def load_state(cls, state):
        job = ParameterSweep(model=state.model, params=state.params)
        keys = cls.__sort_keys(state.results.keys())
        for key in keys:
            job.results[key] = Simulation.load_state(state.results[key])
        return job
    
    def plot_devil_dftd_extinction_over_param(self, res_sub_keys=[], no_plot=False, key=None,
                                              param_label=None, units=None, verbose=False):
        data = self.get_devil_dftd_extinction_over_param(
            res_sub_keys, key=key, return_data=True, verbose=verbose
        )
            
        units = "" if units is None else f" ({units})"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16, 6])
        
        im1 = ax1.plot(data[0], data[1])
        ax1.set_title(f'Dftd elimination vs. {param_label}', fontsize=14)
        ax1.set_ylim(ymin=-1,ymax=101)
        ax1.tick_params(axis="x", labelsize=12)
        ax1.tick_params(axis="y", labelsize=12)
        ax1.set_xlabel(f"{param_label}{units}", fontsize=14)
        ax1.set_ylabel("DFTD elimination probability", fontsize=14)
        
        im2 = ax2.plot(data[0], data[2])
        ax2.set_title(f'Devil extinction vs. {param_label}', fontsize=14)
        ax2.set_ylim(ymin=-1,ymax=101)
        ax2.tick_params(axis="x", labelsize=12)
        ax2.tick_params(axis="y", labelsize=12)
        ax2.set_xlabel(f"{param_label}{units}", fontsize=14)
        ax2.set_ylabel("Devil extinction probability", fontsize=14)

    def run(self, solver=None, params=None, verbose=False):
        self.result_keys = []
        self.simulations = []
        
        if params is not None:
            self.params = params
        
        index = 0
        variables = {}
        self.__load(solver=solver, index=index, variables=variables, verbose=verbose)
        self.__run()