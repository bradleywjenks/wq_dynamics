"""

The following functions are defined in this file:
    - 'load_network_data'
    - 'plot_network'

"""

##### PREAMBLE #####

# import packages
import os
import wntr
import networkx as nx
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

from src.functions import *


# improve matplotlib image quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


# create class for storing data as objects
class WDN(BaseModel):
    A12: Any
    A10: Any
    net_info: dict
    link_df: pd.DataFrame
    node_df: pd.DataFrame
    demand_df: pd.DataFrame
    h0_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


##### FUNCTIONS #####


"""
Load network data via wntr
""" 


def load_network_data(inp_file):
    
    wdn = object()

    # load network from wntr
    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # get network elements and simulation info
    nt = int(wn.options.time.duration / wn.options.time.hydraulic_timestep)
    nt = nt if nt>0 else 1
    net_info = dict(
        np=wn.num_links,
        nn=wn.num_junctions,
        n0=wn.num_reservoirs,
        nt=nt,
        headloss=wn.options.hydraulic.headloss,
        units=wn.options.hydraulic.inpfile_units,
        reservoir_names=wn.reservoir_name_list,
        junction_names=wn.junction_name_list,
        pipe_names=wn.pipe_name_list,
        valve_names=wn.valve_name_list,
        prv_names=wn.prv_name_list
    )

    
    ## extract link data
    if net_info['headloss'] == 'H-W':
        n_exp = 1.852
    elif net_info['headloss'] == 'D-W':
        n_exp = 2

    link_df = pd.DataFrame(
        index=pd.RangeIndex(net_info['np']),
        columns=['link_ID', 'link_type', 'diameter', 'length', 'n_exp', 'C', 'node_out', 'node_in'],
    ) # NB: 'C' denotes roughness or HW coefficient for pipes and local (minor) loss coefficient for valves

    def link_dict(link):
        if isinstance(link, wntr.network.Pipe):  # check if the link is a pipe
            return dict(
                link_ID=link.name,
                link_type='pipe',
                diameter=link.diameter,
                length=link.length,
                n_exp=n_exp,
                C=link.roughness,
                node_out=link.start_node_name,
                node_in=link.end_node_name
            )
        elif isinstance(link, wntr.network.Valve): # check if the link is a valve
            return dict(
                link_ID=link.name,
                link_type='valve',
                diameter=link.diameter,
                length=2*link.diameter,
                n_exp=2,
                C=link.minor_loss,
                node_out=link.start_node_name,
                node_in=link.end_node_name
            )
        
    for idx, link in enumerate(wn.links()):
        link_df.loc[idx] = link_dict(link[1])

    
    # extract node data
    node_df = pd.DataFrame(
        index=pd.RangeIndex(wn.num_nodes), columns=["node_ID", "elev", "xcoord", "ycoord"]
    )

    def node_dict(node):
        if isinstance(node, wntr.network.elements.Reservoir):
            elev = 0
        else:
            elev = node.elevation
        return dict(
            node_ID=node.name,
            elev=elev,
            xcoord=node.coordinates[0],
            ycoord=node.coordinates[1]
        )

    for idx, node in enumerate(wn.nodes()):
        node_df.loc[idx] = node_dict(node[1])


    # compute graph data
    A = np.zeros((net_info['np'], net_info['nn']+net_info['n0']), dtype=int)
    for k, row in link_df.iterrows():
        # find start node
        out_name = row['node_out']
        out_idx = node_df[node_df['node_ID']==out_name].index[0]
        # find end node
        in_name = row['node_in']
        in_idx = node_df[node_df['node_ID']==in_name].index[0]
        
        A[k, out_idx] = -1
        A[k, in_idx] = 1
        
    junction_idx = node_df.index[node_df['node_ID'].isin(net_info['junction_names'])].tolist()
    reservoir_idx = node_df.index[node_df['node_ID'].isin(net_info['reservoir_names'])].tolist()

    A12 = A[:, junction_idx] # link-junction incident matrix
    A10 = A[:, reservoir_idx] # link-reservoir indicent matrix


    # extract demand data
    demand_df = results.node['demand'].T
    col_names = [f'demands_{t}' for t in range(1, len(demand_df.columns)+1)]
    demand_df.columns = col_names
    demand_df.reset_index(drop=False, inplace=True)
    demand_df = demand_df.rename(columns={'name': 'node_ID'})

    if net_info['nt'] > 1:
        demand_df = demand_df.iloc[:, :-1] # delete last time step
        
    demand_df = demand_df[~demand_df['node_ID'].isin(net_info['reservoir_names'])] # delete reservoir nodes


    # extract boundary data
    h0_df = results.node['head'].T
    col_names = [f'h0_{t}' for t in range(1, len(h0_df.columns)+1)]
    h0_df.columns = col_names
    h0_df.reset_index(drop=False, inplace=True)
    h0_df = h0_df.rename(columns={'name': 'node_ID'})

    if net_info['nt'] > 1:
        h0_df = h0_df.iloc[:, :-1] # delete last time step

    h0_df = h0_df[h0_df['node_ID'].isin(net_info['reservoir_names'])] # only reservoir nodes


    # load data to WDN object
    wdn = WDN(
            A12=A12,
            A10=A10,
            net_info=net_info,
            link_df=link_df,
            node_df=node_df,
            demand_df=demand_df,
            h0_df=h0_df,
    )

    return wdn






"""
Plot network function
""" 

def plot_network(wdn, plot_type='layout', prv_nodes=None, afv_nodes=None, dbv_nodes=None, bv_nodes=None, sensor_nodes=None, vals_df=None, t=None, legend_labels=None, sensor_labels=False):

    # unload data
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info
    h0_df = wdn.h0_df
    
    # draw network
    if plot_type == 'layout':
        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

        nx.draw(uG, pos, node_size=0, node_shape='o', edge_color='grey')
        nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=100, node_shape='s', node_color='black', edgecolors='white') # draw reservoir nodes

        if prv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=prv_nodes, node_size=120, node_shape='^', node_color='black', edgecolors='white')

        if bv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=bv_nodes, node_size=65, node_shape='x', linewidths=2, node_color='black', edgecolors='white')
        
        if dbv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=dbv_nodes, node_size=120, node_shape='d', node_color='black', edgecolors='white')
        
        if afv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=afv_nodes, node_size=200, node_shape='*', node_color='black', edgecolors='white')

        if sensor_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=sensor_nodes, node_size=80, node_shape='o', node_color='black', edgecolors='white')

            if sensor_labels:
                sensor_labels = {node: str(idx+1) for (idx, node) in enumerate(sensor_nodes)}
                labels_sen = nx.draw_networkx_labels(uG, pos, sensor_labels, font_size=12, verticalalignment='bottom')
                for _, label in labels_sen.items():
                    label.set_y(label.get_position()[1] + 70)



    elif plot_type == 'hydraulic head':

        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
        
        # create dictionary from dataframe to match node IDs
        vals_df = vals.set_index('node_ID')[f'h_{t}']
        h0_df = h0_df.set_index('node_ID')[f'h0_{t}']

        junction_vals = [vals_df[node] for node in net_info['junction_names']]
        reservoir_vals = [h0_df[node] for node in net_info['reservoir_names']]
        node_vals_all = junction_vals + reservoir_vals

        # color scaling
        min_val = min(node_vals_all)
        max_val = max(node_vals_all)

        # plot hydraulic heads
        cmap = cm.get_cmap('RdYlBu')
        nx.draw(uG, pos, nodelist=net_info['junction_names'], node_size=20, node_shape='o', node_color=junction_vals, cmap=cmap, vmin=min_val, vmax=max_val)
        nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=100, node_shape='s', node_color=reservoir_vals, cmap=cmap, vmin=min_val, vmax=max_val) 
        if prv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=prv_nodes, node_size=100, node_shape='d', node_color='black') # draw pcv nodes (downstream node)

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(node_vals_all)
        colorbar = plt.colorbar(sm)
        colorbar.set_label('Hydraulic head [m]', fontsize=12)

    elif plot_type == 'pressure head':

        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
        
        # create dictionary from dataframe to match node IDs
        vals_df = vals.set_index('node_ID')[f'h_{t}']
        h0_df = h0_df.set_index('node_ID')[f'h0_{t}']

        junction_vals = [vals_df[node] - node_df.loc[node_df['node_ID'] == node, 'elev'].to_numpy()[0] for node in net_info['junction_names']]
        reservoir_vals = [0 for node in net_info['reservoir_names']]
        node_vals_all = junction_vals + reservoir_vals

        # color scaling
        min_val = min(node_vals_all)
        max_val = max(node_vals_all)

        # plot pressure heads
        cmap = cm.get_cmap('RdYlBu')
        nx.draw(uG, pos, nodelist=net_info['junction_names'], node_size=20, node_shape='o', node_color=junction_vals, cmap=cmap, vmin=min_val, vmax=max_val)
        nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=100, node_shape='s', node_color=reservoir_vals, cmap=cmap, vmin=min_val, vmax=max_val) 
        if prv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=prv_nodes, node_size=100, node_shape='d', node_color='black') # draw pcv nodes (downstream node)

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(node_vals_all)
        colorbar = plt.colorbar(sm)
        colorbar.set_label('Pressure head [m]', fontsize=12)

    elif plot_type == 'disinfectant':

            uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
            pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

            cmap = cm.get_cmap('RdYlBu')
            norm = plt.Normalize(vmin=vals_df.iloc[:, t].min(), vmax=vals_df.iloc[:, t].max())
            node_colors = cmap(norm(vals_df.iloc[:, t]))
            nx.draw(uG, pos, nodelist=vals_df.index, node_size=30, node_shape='o', alpha=0.85, linewidths=0, node_color=node_colors, cmap=cmap, edge_color='grey')
            nx.draw_networkx_nodes(uG, pos, nodelist=sensor_nodes, node_size=80, node_shape='o', node_color='black', edgecolors='white') # draw sensor nodes

            # create a color bar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(vals_df.iloc[:, t])
            colorbar = plt.colorbar(sm)
            colorbar.set_label('Disinfectant residual [mg/L]', fontsize=12)
            # colorbar.set_ticks(colorbar_ticks[0])
            # colorbar.set_ticklabels(colorbar_ticks[1], fontsize=11)


    elif plot_type == 'flow':

        edge_df = link_df[['link_ID', 'node_out', 'node_in']]
        edge_df.set_index('link_ID', inplace=True)
        vals_df = vals.set_index('link_ID')[f'q_{t}']
        vals_df = abs(vals_df) * 1000
        edge_df = edge_df.join(vals_df)

        uG = nx.from_pandas_edgelist(edge_df, source='node_out', target='node_in', edge_attr=f'q_{t}')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

        # Define colormap
        cmap = cm.get_cmap('RdYlBu')

        edge_values = nx.get_edge_attributes(uG, f'q_{t}')
        edge_values = list(edge_values.values())

        # color scaling
        norm = plt.Normalize(min(edge_values), max(edge_values))
        edge_colors = cmap(norm(edge_values))


        nx.draw(uG, pos, node_size=0, node_shape='o', node_color='black')
        nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=100, node_shape='s', node_color='black') 
        nx.draw_networkx_edges(uG, pos, edge_color=edge_colors, width=2) 
        if prv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=prv_nodes, node_size=100, node_shape='d', node_color='black') # draw pcv nodes (downstream node)

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(edge_values)
        colorbar = plt.colorbar(sm)
        colorbar.set_label('Flow [L/s]', fontsize=12)       

    
    # # reservoir labels
    # reservoir_labels = {node: '' for node in net_info['reservoir_names']}
    # labels_1 = nx.draw_networkx_labels(uG, pos, reservoir_labels, font_size=11, verticalalignment='bottom')
    # for _, label in labels_1.items():
    #     label.set_y(label.get_position()[1] + 80)

    # # prv labels
    # if prv_nodes is not None:
    #     prv_labels = {node: 'PRV' for node in prv_nodes}
    #     labels_2 = nx.draw_networkx_labels(uG, pos, prv_labels, font_size=11, verticalalignment='bottom')
    #     for _, label in labels_2.items():
    #         label.set_y(label.get_position()[1] + 80)


    # custom legend code
    if legend_labels is not None:
        # legend_labels = {'Inlet (source)': 'black', 'PRV': 'black', 'DBV': 'black', 'BV': 'black', 'AFV': 'black', 'Sensor node': 'blue'}
        legend_handles = [plt.Line2D([0], [0], marker='o' if label == 'Sensor node' else 's' if label == 'Inlet (source)' else '^' if label == 'PRV' else 'd' if label == 'DBV' else 'x' if label == 'BV' else '*' if label == 'AFV' else None, markeredgewidth=2 if label == 'BV' else None, markeredgecolor='black' if label == 'BV' else None, color='white', markerfacecolor=color, markersize=8 if (label == 'Sensor node' or label == 'BV') else 9 if label == 'Inlet (source)' else 10 if (label == 'PRV' or label == 'DBV') else 14 if label == 'AFV' else None, label=label) for label, color in legend_labels.items()]
        plt.legend(handles=legend_handles, loc='upper right', frameon=False)





"""
Plot sensor values spatially
"""

def plot_sensor_data(wdn, sensor_nodes, vals, legend_labels=None, sensor_labels=False):

    # unload data
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info
    h0_df = wdn.h0_df

    # draw network
    uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
    pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
    nx.draw(uG, pos, node_size=0, node_shape='o', node_color='black')

    # draw sensor nodes
    nx.draw_networkx_nodes(uG, pos, sensor_nodes, node_size=100, node_shape='o', node_color='black', edgecolors='white')

    # plot sensor vals
    if vals is not None:

        cmap = cm.get_cmap('RdYlBu').reversed()

        # get data
        sensor_vals = vals.to_numpy()

        # plot residuals
        nx.draw_networkx_nodes(uG, pos, nodelist=sensor_nodes, node_size=100, node_shape='o', node_color=sensor_vals, cmap=cmap, edgecolors='white')

        # create color bar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(vals)
        colorbar = plt.colorbar(sm)

        if vals.columns[0] == 'cv':
            colorbar.set_label('Coefficient of variation', fontsize=12)
        elif vals.columns[0] == 'sd':
            colorbar.set_label('Standard deviation [mg/L]', fontsize=12)

    if sensor_labels:
        sensor_labels = {node: str(idx+1) for (idx, node) in enumerate(sensor_nodes)}
        labels_sen = nx.draw_networkx_labels(uG, pos, sensor_labels, font_size=12, verticalalignment='bottom')
        for _, label in labels_sen.items():
            label.set_y(label.get_position()[1] + 70)

    # custom legend code
    if legend_labels is not None:
        # legend_labels = {'Inlet (source)': 'black', 'PRV': 'black', 'DBV': 'black', 'BV': 'black', 'AFV': 'black', 'Sensor node': 'blue'}
        legend_handles = [plt.Line2D([0], [0], marker='o' if label == 'Sensor node' else 's' if label == 'Inlet (source)' else '^' if label == 'PRV' else 'd' if label == 'DBV' else 'x' if label == 'BV' else '*' if label == 'AFV' else None, markeredgewidth=2 if label == 'BV' else None, markeredgecolor='black' if label == 'BV' else None, color='white', markerfacecolor=color, markersize=8 if (label == 'Sensor node' or label == 'BV') else 9 if label == 'Inlet (source)' else 10 if (label == 'PRV' or label == 'DBV') else 14 if label == 'AFV' else None, label=label) for label, color in legend_labels.items()]
        plt.legend(handles=legend_handles, loc='upper right', frameon=False)







"""
Set controls function
""" 

def set_controls(net_name, data_path, scenario, bv_close=None, bv_open=None, prv=None, prv_dir=None, dbv=None, afv=None, sim_days=1):

    # BV data
    bv_open_setting = 1e-04
    bv_close_setting = 1e10

    # load dynamic set-points
    if scenario == 'self-cleaning control':
        prv_setting = pd.read_csv(os.path.join(data_path, 'prv_scc_settings.csv'))
    else: 
        prv_setting = pd.read_csv(os.path.join(data_path, 'prv_exist_settings.csv'))

    prv_setting = np.tile(prv_setting, sim_days) 
    dbv_setting = pd.read_csv(os.path.join(data_path, 'dbv_exist_settings.csv'))
    dbv_setting = np.tile(dbv_setting, sim_days)
    afv_setting = pd.read_csv(os.path.join(data_path, 'afv_scc_settings.csv'))
    # afv_setting = np.tile(afv_setting, sim_days)
    afv_time = np.arange(38, 42)

    # load network data
    wdn = load_network_data(os.path.join(data_path, net_name))

    # load network via wntr
    net_path = os.path.join(data_path, net_name)
    wn = wntr.network.WaterNetworkModel(net_path)

    if bv_close is not None:   
        # assign closed BV settings
        for idx, name in enumerate(bv_close):
            link_data = wn.get_link(name)
            link_data = wn.get_link(name).initial_setting = bv_close_setting
            link_data = wn.get_link(name).initial_status = "Active"

    if bv_open is not None:   
        # assign open BV settings
        for idx, name in enumerate(bv_open):
            link_data = wn.get_link(name)
            link_data = wn.get_link(name).initial_setting = bv_open_setting
            link_data = wn.get_link(name).initial_status = "Active"
    
    if dbv is not None:
        # assign initial DBV settings
        for idx, name in enumerate(dbv):
            link_data = wn.get_link(name)
            link_data = wn.get_link(name).initial_setting = dbv_setting[idx, 1] # initial control
            link_data = wn.get_link(name).initial_status = "Active"
        
        # assign time-based DBV controls
        dbv_controls = []
        for t in np.arange(wdn.net_info['nt'] * sim_days):
            for (idx, name) in enumerate(dbv):
                valve = wn.get_link(name)
                dbv_controls = name + "_control_setting_t_" + str(t/4)
                cond = wntr.network.controls.SimTimeCondition(wn, "=", t*900)
                act = wntr.network.controls.ControlAction(valve, "setting", dbv_setting[idx, t])
                rule = wntr.network.controls.Rule(cond, [act], name=dbv_controls)
                wn.add_control(dbv_controls, rule)

    if prv is not None:       
        # assign initial PRV settings
        for idx, name in enumerate(prv):
            link_data = wn.get_link(name)
            wn.remove_link(name)
            if prv_dir[idx] == 1:
                wn.add_valve(name, link_data.start_node_name, link_data.end_node_name, diameter=link_data.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_setting[idx, 0], initial_status="Active")
            else:
                wn.add_valve(name, link_data.end_node_name, link_data.start_node_name, diameter=link_data.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_setting[idx, 0], initial_status="Active")
        
        # assign time-based PRV controls
        prv_controls = []
        for t in np.arange(wdn.net_info['nt'] * sim_days):
            for (idx, name) in enumerate(prv):
                valve = wn.get_link(name)
                prv_controls = name + "_prv_setting_t_" + str(t/4)
                cond = wntr.network.controls.SimTimeCondition(wn, "=", t*900)
                act = wntr.network.controls.ControlAction(valve, "setting", prv_setting[idx, t])
                rule = wntr.network.controls.Rule(cond, [act], name=prv_controls)
                wn.add_control(prv_controls, rule)
    
    if afv is not None:
        # assign AFV demands
        afv_controls = np.zeros((wdn.net_info['nt'], len(afv)))
        afv_controls[afv_time, :] = afv_setting
        sim_time = np.arange(0, wn.options.time.duration, 900)

        afv_df = pd.DataFrame(afv_controls, columns=afv, index=sim_time)
        wn.assign_demand(afv_df, pattern_prefix="afv_demand")

    return wn




"""
Plot temporal metric
""" 

def plot_temporal_metric(wdn, temporal_metric, df_flow, df_trace, sensor_names, sim_days_hyd=1):

    # unload data
    link_df = wdn.link_df
    node_df = wdn.node_df
        
    if temporal_metric == 'flow reversal':
        
        # metric data
        q = df_flow.to_numpy().T
        metric = pd.DataFrame(index=df_flow.T.index)
        metric['rev_count'] = 0

        for j in np.arange(wdn.net_info['np']):
            a = 0
            for k in np.arange(1, wdn.net_info['nt']*sim_days_hyd):
                if np.abs(q[j, k-1]) > 1e-3 and np.abs(q[j, k]) > 1e-3:
                    if np.sign(q[j, k-1]) * np.sign(q[j, k]) == -1:
                        a += 1
            
            metric['rev_count'][j] = a
            
        edge_weight_name = 'rev_count'
        cbar_title = "Flow reversal count"
        colorbar_ticks = (np.arange(0, 6, 1), [str(int(x)) for x in np.arange(0, 5, 1)] + [r"$\geq 5$"])
        clims = (0, 5)
        cmap = cm.Reds

        # draw network and plot edge weights
        edge_weights = link_df[['link_ID', 'node_out', 'node_in']]
        edge_weights.set_index('link_ID', inplace=True)
        edge_weights = edge_weights.join(metric)
        uG = nx.from_pandas_edgelist(edge_weights, source='node_out', target='node_in', edge_attr=edge_weight_name)
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
        
        edge_values = nx.get_edge_attributes(uG, edge_weight_name)
        edge_values = list(edge_values.values())
        norm = plt.Normalize(vmin=clims[0], vmax=clims[1])
        edge_colors = cmap(norm(edge_values))
        
        flow_cv_plot = nx.draw(uG, pos, node_size=0, node_shape='o', node_color='black')
        flow_cv_plot = nx.draw_networkx_edges(uG, pos, edge_color=edge_colors, width=2) 
        flow_cv_plot = nx.draw_networkx_nodes(uG, pos, nodelist=sensor_names, node_size=80, node_shape='o', node_color='black', edgecolors='white') # draw sensor nodes

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(edge_values)
        colorbar = plt.colorbar(sm)
        colorbar.set_label(cbar_title, fontsize=12)
        colorbar.set_ticks(colorbar_ticks[0])
        colorbar.set_ticklabels(colorbar_ticks[1], fontsize=11)
        
        legend_labels = {'Sensor node': 'black'}
        legend_handles = [plt.Line2D([0], [0], marker='o', markeredgewidth=2, markeredgecolor='white', color='white', markerfacecolor=color, markersize=10, label=label)  for label, color in legend_labels.items()]
        plt.legend(handles=legend_handles, loc='upper right', frameon=False)
        
    elif temporal_metric == 'flow cv':
        
        # metric data
        metric = pd.DataFrame((df_flow.std() / df_flow.mean()).abs(), columns=['flow_cv'])
        edge_weight_name = 'flow_cv'
        cbar_title = 'Flow CV'
        colorbar_ticks = (np.arange(0.4, 1.5, 0.2), [r"$<0.4$"] + [str(round(x,2)) for x in np.arange(0.6, 1.3, 0.2)] + [r"$\geq 1.4$"])
        clims = (0.4, 1.4)
        cmap = cm.Blues

        # draw network and plot edge weights
        edge_weights = link_df[['link_ID', 'node_out', 'node_in']]
        edge_weights.set_index('link_ID', inplace=True)
        edge_weights = edge_weights.join(metric)
        uG = nx.from_pandas_edgelist(edge_weights, source='node_out', target='node_in', edge_attr=edge_weight_name)
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
        
        edge_values = nx.get_edge_attributes(uG, edge_weight_name)
        edge_values = list(edge_values.values())
        norm = plt.Normalize(vmin=clims[0], vmax=clims[1])
        edge_colors = cmap(norm(edge_values))
        
        flow_rev_plot = nx.draw(uG, pos, node_size=0, node_shape='o', node_color='black')
        flow_rev_plot = nx.draw_networkx_edges(uG, pos, edge_color=edge_colors, width=2) 
        flow_rev_plot = nx.draw_networkx_nodes(uG, pos, nodelist=sensor_names, node_size=80, node_shape='o', node_color='black', edgecolors='white') # draw sensor nodes

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(edge_values)
        colorbar = plt.colorbar(sm)
        colorbar.set_label(cbar_title, fontsize=12)
        colorbar.set_ticks(colorbar_ticks[0])
        colorbar.set_ticklabels(colorbar_ticks[1], fontsize=11)
        
        legend_labels = {'Sensor node': 'black'}
        legend_handles = [plt.Line2D([0], [0], marker='o', markeredgewidth=2, markeredgecolor='white', color='white', markerfacecolor=color, markersize=10, label=label)  for label, color in legend_labels.items()]
        plt.legend(handles=legend_handles, loc='upper right', frameon=False)
        
    elif temporal_metric == 'source trace':
        
        # metric data
        metric = df_trace.T
        node_weight_name = 'source_trace'
        cbar_title = 'Mean source trace [%]'
        colorbar_ticks = (np.arange(50, 101, 10), [r"$<50$"] + [str(int(x)) for x in np.arange(60, 101, 10)])
        clims = (50, 100)
        cmap = cm.get_cmap('RdYlBu')

        # draw network and plot node weights
        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
        
        norm = plt.Normalize(vmin=clims[0], vmax=clims[1])
        node_colors = cmap(norm(metric[node_weight_name]))
        nx.draw(uG, pos, nodelist=metric.index, node_size=30, node_shape='o', alpha=0.85, linewidths=0, node_color=node_colors, cmap=cmap, edge_color='grey')
        nx.draw_networkx_nodes(uG, pos, nodelist=sensor_names, node_size=80, node_shape='o', node_color='black', edgecolors='white') # draw sensor nodes

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(metric[node_weight_name])
        colorbar = plt.colorbar(sm)
        colorbar.set_label(cbar_title, fontsize=12)
        colorbar.set_ticks(colorbar_ticks[0])
        colorbar.set_ticklabels(colorbar_ticks[1], fontsize=11)
        
        legend_labels = {'Sensor node': 'black'}
        legend_handles = [plt.Line2D([0], [0], marker='o', markeredgewidth=2, markeredgecolor='white', color='white', markerfacecolor=color, markersize=10, label=label)  for label, color in legend_labels.items()]
        plt.legend(handles=legend_handles, loc='upper right', frameon=False)

