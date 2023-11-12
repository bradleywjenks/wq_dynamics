"

Plot sensor nodes and their connection strength

Steps:
    1. Load network with OpWater and solve network hydraulics
    2. Input sensor node names and find index values
    3. Plot sensor node connections
    4. Find unique paths and highlight
        - Observable paths
        - Unobservable paths

"

using Revise
using GraphRecipes, Plots
using PyCall
using DataFrames
using GraphPlot
using Graphs
using Colors
using ColorSchemes
using CSV
using PyPlot
using StatsModels
using Combinatorics
pyplot()

begin
    @pyimport wntr
    @pyimport pandas as pd
    @pyimport statsmodels.tsa.stattools as stattools
end

# input data
begin
    net_name = "bwfl_2022_05_hw"

    net_path = "/home/bradw/workspace/networks/data/"
    data_path = "/home/bradw/workspace/wq_temporal_connectivity/data/"
    
end


# # import network data from OpWater
# begin
#     net_dir = "bwfl_2022_05/hw"
#     network = load_network(net_dir, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=true)
# end


# create water network model
begin
    net_dir = "bwfl_2022_05/hw"
    inp_file = net_path * net_dir * "/" * net_name * ".inp"

    wn = wntr.network.WaterNetworkModel(inp_file)

    # store original network element names
    reservoir_names = wn.reservoir_name_list
    junction_names = wn.junction_name_list
    link_names = wn.link_name_list
    pipe_names = wn.pipe_name_list
    valve_names = wn.valve_name_list

end

# sensor data
begin
    sensor_names = ["node_2746", "node_1811", "node_2367", "node_2747", "node_1809", "node_0187", "node_2506", "node_1773", "node_1802"] # edit accordingly
    # sensor_idx = [findfirst(name -> name == sensor, vcat(junction_names, reservoir_names)) for sensor in sensor_names]
end


# load chlorine time series data (already in order of sensor names)
wq_df = CSV.read(data_path * "wq_data.csv", DataFrame)

# check sensor connectivity
begin

    # insert code here...
    
end


# compute granger causality for each sensor configuration
begin

    nC_tot = 36  # replace with function to compute number of combinations
    max_lag = 24 * 4 # replace with water age results
    granger_results = zeros(nC_tot * 2, max_lag)
    sensor_comb = Array{Int}(zeros(nC_tot * 2, 2))
    nC = 0

    for (sensor_1, sensor_2) in combinations(sensor_names, 2)

        # get sensor indices
        idx_1 = findall(x->x==sensor_1, sensor_names)[1]
        idx_2 = findall(x->x==sensor_2, sensor_names)[1]

        # results for direction 1
        nC += 1
        data = Matrix(wq_df[!, [sensor_2, sensor_1]])
        Plots.plot(data)
        sensor_comb[nC, :] .= idx_1, idx_2
        stationary_results = stattools.adfuller(data[:, 1])
        granger_results = stattools.grangercausalitytests(data, max_lag)


        # results for direction 2
        nC += 1
        data = Matrix(wq_df[!, [sensor_2, sensor_1]])
        sensor_comb[nC, :] .= idx_2, idx_1
        for t in collect(1:n_lag)
            # compute granger causality
            granger_results[nC, t] = 0.1 # replace with function to get results
        end

    end

end

begin

    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", family="CMU Serif")
    PyPlot.rc("figure", dpi=300)
    PyPlot.rc("savefig", dpi=300)

    x = collect(0:0.25:47.75)
    # y_1 = Matrix(wq_df[!, [sensor_names[1]]])[1:192]
    y_2 = Matrix(wq_df[!, [sensor_names[6]]])[1:192]
    cl = Plots.plot()
    # cl = Plots.plot!(x, y_1, label="", linecolor=:blue, linewidth=1.5)
    cl = Plots.plot!(x, y_2, label="", linecolor=:blue, linewidth=1.5)
    cl = Plots.plot!(xlabel="Time [h]", ylabel="Chlorine [mg/L]", xticks=(0:12:48), xtickfontsize=18, ytickfontsize=18, xguidefontsize=20, yguidefontsize=20, ylim=(0, 0.8))

end