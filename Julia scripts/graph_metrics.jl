"

Find all paths between sensor nodes:
Steps:
    1. Load network with OpWater and solve network hydraulics wth WNTR
    2. Input sensor node names and find index values
    3. For loop in range(1:nt)
        - Form directed graph based on flow direction at each time Steps
        - Find and save all paths between sesnor nodes
    4. Find unique paths and highlight
        - Observable paths
        - Unobservable paths

"

# load dependencies
using Revise
using OpWater
using GraphRecipes, Plots
using JuMP
using PyCall
using Random
using Distributions
using DataFrames
using GraphPlot
using SparseArrays
using Graphs
using Combinatorics
using Colors
using ColorSchemes
using CSV
using PyPlot
using LaTeXStrings
pyplot()

begin
    @pyimport wntr
    @pyimport pandas as pd
end


# input data
begin
    net_name = "bwfl_2022_05_hw"
    # net_name = "bwfl_2022_05_hw_control"
    # net_name = "pescara"

    net_path = "/home/bradw/workspace/networks/data/"
    data_path = "/home/bradw/workspace/wq_temporal_connectivity/data/"

    # inputs from optimisation results
    prv_new_idx = [1963, 749, 1778]
    prv_new_dir = [-1, -1, -1] 
    afv_new_idx = [1285, 1236, 2444, 2095]

    # inputs from existing controls
    prv_exist_idx = [2755, 2728, 2742]
    prv_exist_dir = [1, 1, 1] 
    dbv_idx = [2746, 2747]

    # boundary valves
    bv_open_idx = [2388, 2554, 2507, 2235]
    bv_close_idx = [2430]

end

### STEP 1: LOAD NETWORK AND SIMULATE HYDRAULICS ###

# create network data from OpWater
begin

    if net_name == "bwfl_2022_05_hw" || net_name == "bwfl_2022_05_hw_control"
        net_dir = "bwfl_2022_05/hw"
        network = load_network(net_dir, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=true)
    elseif net_name == "BWFLnet_SS"
        net_dir = "bwfl"
        network = load_network(net_dir, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=false)
    else
        network = load_network(net_name, afv_idx=false, dbv_idx=false, pcv_idx=false)
    end

end

# create water network model
begin

    if net_name == "bwfl_2022_05_hw" || net_name == "bwfl_2022_05_hw_control"
        net_dir = "bwfl_2022_05/hw"
        inp_file = net_path * net_dir * "/" * net_name * ".inp"
    elseif net_name == "BWFLnet_SS"
        net_dir = "bwfl"
        inp_file = net_path * net_dir * "/" * net_name * ".inp"
    else
        inp_file = net_path * net_name * "/" * net_name * ".inp"
    end

    wn = wntr.network.WaterNetworkModel(inp_file)

    # store original network element names
    reservoir_names = wn.reservoir_name_list
    junction_names = wn.junction_name_list
    link_names = wn.link_name_list
    node_names = wn.node_name_list
    pipe_names = wn.pipe_name_list
    valve_names = wn.valve_name_list

end

# select number of simulation days
nt_d = 10

# assign controls
begin

    scenario = "existing control"
    bv_open_setting = 1e-04 # open BVs
    bv_close_setting = 1e10 # closed BVs

    # CASE 1: all links open
    if scenario == "open"

        # bv link data
        bv_names = link_names[vcat(bv_open_idx, bv_close_idx)]

        # set bv settings
        for (idx, b) ∈ enumerate(bv_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting = bv_open_setting

        end

    # CASE 2: separated DMAs (closed BVs)
    elseif scenario == "separated"

        # dbv and bv link data
        dbv_names = link_names[dbv_idx]
        bv_open_names = link_names[bv_open_idx]
        bv_close_names = link_names[bv_close_idx]

        # set dbv settings
        for (idx, b) ∈ enumerate(dbv_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting = bv_close_setting
            link_data = wn.get_link(b).initial_status = "Active"

        end

        # set open bv settings
        for (idx, b) ∈ enumerate(bv_open_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting = bv_open_setting
            link_data = wn.get_link(b).initial_status = "Active"

        end

        # set close bv settings
        for (idx, b) ∈ enumerate(bv_close_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting = bv_close_setting
            link_data = wn.get_link(b).initial_status = "Active"

        end


    elseif scenario == "existing control"

        # prv_exist, dbv, and bv link data
        prv_names = link_names[prv_exist_idx]
        prv_dir = prv_exist_dir
        dbv_names = link_names[dbv_idx]
        bv_open_names = link_names[bv_open_idx]
        bv_close_names = link_names[bv_close_idx]

        dbv_setting = CSV.File(data_path * "dbv_exist_settings.csv") |> Tables.matrix
        prv_setting = CSV.File(data_path * "prv_exist_settings.csv") |> Tables.matrix

        if nt_d > 1
            dbv_setting = repeat(dbv_setting, 1, nt_d)
            prv_setting = repeat(prv_setting, 1, nt_d)
        end

        # set initial dbv settings
        for (idx, b) ∈ enumerate(dbv_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting=dbv_setting[idx, 1]
            link_data = wn.get_link(b).initial_status = "Active"
    
        end

        # add time-based dbv controls
        dbv_controls = nothing
        for t ∈ collect(1:size(dbv_setting, 2))
            for (idx, d) ∈ enumerate(dbv_names)
                valve = wn.get_link(d)
                dbv_controls = d * "_control_setting_t_" * string(t)
                cond = wntr.network.controls.SimTimeCondition(wn, "=", t*900)
                act = wntr.network.controls.ControlAction(valve, "setting", dbv_setting[idx, t])
                rule = wntr.network.controls.Rule(cond, [act], name=dbv_controls)
                wn.add_control(dbv_controls, rule)
            end
        end

        # set initial prv settings
        for (idx, p) ∈ enumerate(prv_names)

            link_data = wn.get_link(p)

            wn.remove_link(p)
            if prv_dir[idx] == 1
                wn.add_valve(p, link_data.start_node_name, link_data.end_node_name, diameter=link_data.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_setting[idx, 1], initial_status="Active")
            else
                wn.add_valve(p, link_data.end_node_name, link_data.start_node_name, diameter=link_data.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_setting[idx, 1], initial_status="Active")
            end

        end

        # add time-based prv controls
        prv_controls = nothing
        for t ∈ collect(1:size(prv_setting, 2))
            for (idx, p) ∈ enumerate(prv_names)
                valve = wn.get_link(p)
                prv_controls = p * "_prv_setting_t_" * string(t)
                cond = wntr.network.controls.SimTimeCondition(wn, "=", t*900)
                act = wntr.network.controls.ControlAction(valve, "setting", prv_setting[idx, t])
                rule = wntr.network.controls.Rule(cond, [act], name=prv_controls)
                wn.add_control(prv_controls, rule)
            end
        end

        # set open bv settings
        for (idx, b) ∈ enumerate(bv_open_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting = bv_open_setting

        end

        # set close bv settings
        for (idx, b) ∈ enumerate(bv_close_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting = bv_close_setting

        end


    elseif scenario == "scc control"

        # prv_new, afv_new, and bv link data
        prv_names = link_names[prv_new_idx]
        prv_dir = prv_new_dir
        afv_names = junction_names[afv_new_idx]
        bv_open_names = link_names[vcat(bv_open_idx, bv_close_idx)]

        afv_setting = CSV.read(data_path * "afv_scc_settings.csv", Tables.matrix)
        prv_setting = CSV.read(data_path * "prv_scc_settings.csv", Tables.matrix)

        if nt_d > 1
            afv_setting = repeat(afv_setting, 1, nt_d)
            prv_setting = repeat(prv_setting, 1, nt_d)
        end


        # set initial prv settings
        for (idx, p) ∈ enumerate(prv_names)

            link_data = wn.get_link(p)

            wn.remove_link(p)
            if prv_dir[idx] == 1
                wn.add_valve(p, link_data.start_node_name, link_data.end_node_name, diameter=link_data.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_setting[idx, 1], initial_status="Active")
            else
                wn.add_valve(p, link_data.end_node_name, link_data.start_node_name, diameter=link_data.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_setting[idx, 1], initial_status="Active")
            end

        end

        # add time-based prv controls
        prv_controls = nothing
        for t ∈ collect(1:size(prv_setting, 2))
            for (idx, p) ∈ enumerate(prv_names)
                valve = wn.get_link(p)
                prv_controls = p * "_prv_setting_t_" * string(t)
                cond = wntr.network.controls.SimTimeCondition(wn, "=", t*900)
                act = wntr.network.controls.ControlAction(valve, "setting", prv_setting[idx, t])
                rule = wntr.network.controls.Rule(cond, [act], name=prv_controls)
                wn.add_control(prv_controls, rule)
            end
        end


        # flushing demands at AFV nodes
        α_time = collect(38:42)                    # Input from optimization results
        α = zeros(network.nt, size(afv_new_idx, 1))
        α[α_time, :] .= afv_setting
        sim_time = collect(0:900:(wn.options.time.duration)-900)

        df = pd.DataFrame(α, columns=afv_names, index=sim_time)
        wn.assign_demand(df, pattern_prefix="flush_demands")

        # set open bv settings
        for (idx, b) ∈ enumerate(bv_open_names)

            link_data = wn.get_link(b)
            link_data = wn.get_link(b).initial_setting = bv_open_setting

        end

    end

end

# simulate network hydraulics & water quality
begin

    # simulation times
    nt_h = 23.75
    wn.options.time.duration = nt_d * nt_h * 3600
    wn.options.time.hydraulic_timestep = 60 * 15
    wn.options.time.report_timestep = 60 * 15
    # wn.options.time.duration = 23 * 3600
    nt = Int(wn.options.time.duration ./ wn.options.time.hydraulic_timestep)
    wn.options.time.rule_timestep = wn.options.time.hydraulic_timestep
    wn.convert_controls_to_rules(priority=3) # convert controls to rules

    # setup water quality parameters
    wn.options.quality.parameter = "CHEMICAL"
    wn.options.reaction.bulk_coeff = -2.5e-05 # units = 1/second
    wn.options.reaction.wall_coeff = 0.0

    source_node_1 = reservoir_names[1]
    d_1 = Truncated(Normal(0.5, 0.0), 0, Inf)
    source_pattern_1 = rand(d_1, Int(nt/nt_d))
    wn.add_pattern("source_pattern_1", source_pattern_1)
    wn.add_source("source_1", source_node_1, "CONCEN", 1, "source_pattern_1")

    source_node_2 = reservoir_names[2]
    d_2 = Truncated(Normal(1.5, 0.0), 0, Inf)
    source_pattern_2 = rand(d_2, Int(nt/nt_d))
    wn.add_pattern("source_pattern_2", source_pattern_2)
    wn.add_source("source_2", source_node_2, "CONCEN", 1, "source_pattern_2")


    # run solver
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # obtain flow at links
    df_flow = DataFrame()
    df_flow[!, "timestamp"] = results.link["flowrate"].index.values ./ 3600
    for col ∈ results.link["flowrate"].columns
        df_flow[!, col] = getproperty(results.link["flowrate"], col).values .* 1000
    end

    # obtain pressure at nodes
    df_pressure = DataFrame()
    df_pressure[!, "timestamp"] = results.node["pressure"].index.values ./ 3600
    for col ∈ results.node["pressure"].columns
        df_pressure[!, col] = getproperty(results.node["pressure"], col).values
    end

    # obtain quality at values
    df_qual = DataFrame()
    df_qual[!, "timestamp"] = results.node["quality"].index.values ./ 3600
    for col ∈ results.node["quality"].columns
        df_qual[!, col] = getproperty(results.node["quality"], col).values #./ 3600 # covnert to water age to hours
    end

end

# sort incidence matrix A based on flow direction from hydraulic results
begin

    # node_to_index = [node_name: i for (i, node_name) in enumerate(wn.node_name_list)]
    node_to_index = Dict(wn.node_name_list[i] => i for i in 1:length(wn.node_name_list))

    # Create an empty incidence matrix
    num_nodes = length(wn.node_name_list)
    num_links = length(wn.link_name_list)
    A = zeros(num_nodes, num_links)

    # Populate the incidence matrix
    for (link_idx, link_name) in enumerate(wn.link_name_list)
        link = wn.get_link(link_name)
        start_node_idx = node_to_index[link.start_node_name]
        end_node_idx = node_to_index[link.end_node_name]
        A[start_node_idx, link_idx] = -1
        A[end_node_idx, link_idx] = 1
    end
    A = A'
    
    # initialise A matrix
    # A = hcat(Matrix(network.A12), Matrix(network.A10))
    A_temp = zeros(size(A, 1), size(A, 2), size(df_flow, 1))
    for t in collect(1:network.nt)
        A_temp[:, :, t] .= A
    end

    # make directed graph at each time step
    q = Matrix(df_flow[:, 2:end])'
    for t in collect(1:network.nt)
        for j in collect(1:network.np)
            w = findall(x -> x == -1, A_temp[j, :, t])
            z = findall(x -> x == 1, A_temp[j, :, t])
            if q[j, t] < 0
                A_temp[j, w, t] .= 1
                A_temp[j, z, t] .= -1
            end
            # if j in prv_exist_idx
            #     A_temp[j, w, t] .= 1
            #     A_temp[j, z, t] .= -1
            # end
        end
    end

end


### STEP 2: FIND SENSOR NODE INDICES ###
begin

    reservoir_names = wn.reservoir_name_list
    junction_names = wn.junction_name_list
    node_names = wn.node_name_list
    link_names = wn.link_name_list
    pipe_names = wn.pipe_name_list
    valve_names = wn.valve_name_list

    sensor_names = ["node_2746", "node_1811", "node_2367", "node_2747", "node_1809", "node_0187", "node_2506", "node_1773", "node_1802"] # edit accordingly
    # sensor_names = ["42", "57"]
    sensor_idx = [findfirst(name -> name == sensor, wn.node_name_list) for sensor in sensor_names]
end


### STEP 3: FIND OBSERVABLE PATHS IN NETWORK ###

all_paths = Dict{Tuple, Vector{Vector{Int}}}()

# define all_simple_paths function
function all_simple_paths(graph::DiGraph, start_node, end_node)
    paths = Vector{Vector{Int}}()
    current_path = Vector{Int}()

    function dfs(node)
        push!(current_path, node)

        if node == end_node
            push!(paths, copy(current_path))
        else
            for neighbor in outneighbors(graph, node)
                if neighbor ∉ current_path
                    dfs(neighbor)
                end
            end
        end

        pop!(current_path)
    end

    dfs(start_node)
    return paths
end

begin

    for t in collect(1:network.nt)

        # initialise adjacency matrix
        iA = Int64[]
        jA = Int64[]
        vA = Float64[]
        num_junctions = network.nn + network.n0
        adj = spzeros(num_junctions, num_junctions)

        for j in collect(1:network.np)
            node_in = findall(A_temp[j, :, t] .> 0)[1]
            node_out = findall(A_temp[j, :, t] .< 0)[1]
            # node_in, node_out = node_in[1], node_out[1]
            adj[node_out, node_in] = 1
            # adj[node_out, node_in] = -1
        end

        # create a directed graph for the current time step using A12_temp[:, :, t]
        G = SimpleDiGraph(sparse(adj))

        for (source_node, target_node) in combinations(sensor_idx, 2)

            # Find all simple paths between the current source and target node
            paths = all_simple_paths(G, source_node, target_node)
            
            # Store the paths in the dictionary
            all_paths[(t, source_node, target_node)] = paths
        end

    end


    # collect all unique nodes in all_simple_paths
    all_nodes = Set([node for (key, paths) in all_paths for path in paths for node in path]) 

end


### STEP 4: HIGHLIGHT OBSERVABLE PATHS IN NETWORK ###

begin

    # make graph again
    iA = Int64[]
    jA = Int64[]
    vA = Float64[]
    num_junctions = network.nn + network.n0
    adj = spzeros(num_junctions, num_junctions)

    for j in collect(1:network.np)
        node_in = findall(A[j, :] .> 0)[1]
        node_out = findall(A[j, :] .< 0)[1]
        # node_in, node_out = node_in[1], node_out[1]
        adj[node_out, node_in] = 1
        adj[node_in, node_out] = 1
    end
    G = SimpleGraph(sparse(adj))
    edge_vals = zeros(network.np)
    line_width = 1 * ones(network.np)
    for (i, edge) in enumerate(edges(G))

        if src(edge) in all_nodes && dst(edge) in all_nodes
            edge_vals[i] = 1
            line_width[i] = 2
        end

    end
end

# make graph plot
begin

    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", family="CMU Serif")
    PyPlot.rc("figure", dpi=300)
    PyPlot.rc("savefig", dpi=300)

    # edge_colour = palette([:grey85, ColorSchemes.Reds[6]])
    # edge_colour = palette([:grey85, ColorSchemes.OrRd[6]])
    edge_colour = palette([:grey85, ColorSchemes.Blues[6]])

    # line_colour = ColorSchemes.Reds[6]
    # line_colour = ColorSchemes.OrRd[6]
    line_colour = ColorSchemes.Blues[6]

    # get x and y positions
    x_coord = zeros(wn.num_nodes)
    y_coord = zeros(wn.num_nodes)
    for (name, node) in wn.nodes()
        idx = findfirst(x->x == name, node_names)
        x_coord[idx] = node.coordinates[1]
        y_coord[idx] = node.coordinates[2]
    end

    pos_x = (2 * x_coord .- (minimum(x_coord) + maximum(x_coord))) ./ (maximum(x_coord) - minimum(x_coord))
    pos_y = (-2 * y_coord .- (maximum(y_coord) + minimum(y_coord))) ./ (minimum(y_coord) - maximum(y_coord))

    fig = Plots.plot()  

    # plot edges
    edge_x = []
    edge_y = []
    for (idx, edge) in enumerate(edges(G))
        push!(edge_x, pos_x[src(edge)])
        push!(edge_x, pos_x[dst(edge)])
        push!(edge_x, NaN)
        push!(edge_y, pos_y[src(edge)])
        push!(edge_y, pos_y[dst(edge)])
        push!(edge_y, NaN)
    end
    fig = Plots.plot(edge_x, edge_y, frame_style=:none, linecolor=line_colour, linewidth=0.5, label="Observable path")

    # plot observable paths
    for (idx, edge) in enumerate(edges(G))
        fig = Plots.plot!([pos_x[src(edge)], pos_x[dst(edge)]], [pos_y[src(edge)], pos_y[dst(edge)]], line_z=edge_vals[idx], linecolor=edge_colour, linewidth=line_width[idx], label="", linealpha=1, colorbar=:none)
    end


    # plot sensor nodes
    sen_x = []
    sen_y = []
    for sen in sensor_idx
        push!(sen_x, pos_x[sen])
        push!(sen_y, pos_y[sen])
    end
    fig = Plots.scatter!(sen_x, sen_y, markershape=:circle, c=:black, markerstrokecolor=:white, markerstrokewidth=1, markeralpha=1, markersize=8, label="Sensor node")

    # # plot reservoir nodes
    # reservoir_x = []
    # reservoir_y = []
    # for reservoir in network.reservoir_idx
    #     push!(reservoir_x, pos_x[reservoir])
    #     push!(reservoir_y, pos_y[reservoir])
    # end
    # fig = Plots.scatter!(reservoir_x, reservoir_y, markershape=:square, c=:black, markerstrokecolor=:white, markerstrokewidth=1, markeralpha=1, markersize=6, label="Inlet")

    # plot formatting
    fig = Plots.plot!(legendfont=12, titlefont=14, colorbar_tickfontsize=12, colorbar_titlefontsize=14, frame_style=:none, showaxis=false, grid=false, aspect_ratio=:equal)

end



### STEP 5: PLOT DIFFERENT FLOW INDICATORS ###

# flow range as edge values
begin

    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", family="CMU Serif")
    PyPlot.rc("figure", dpi=300)
    PyPlot.rc("savefig", dpi=300)

    # flow_metric = "flow_rev"
    flow_metric = "range"

    if flow_metric == "range"
        # flow range
        q_range = vec(mapslices(range -> (maximum(range) - minimum(range)), q, dims=2))
        edge_vals = q_range
        min_val = minimum(q_range)
        max_val = maximum(q_range)
        q_range_norm = (q_range .- min_val) ./ (max_val - min_val)
        # edge_vals = q_range_norm
        cbar_title = "Flow range [L/s]"
        colorbar_ticks = (0:5:25, [string.(round.(Int, (0:5:20))); L"$\geq 25$"])
        clims = (0, 25)
        # colorbar_ticks = (0:0.2:1)

    elseif flow_metric == "peak"
        # peak flow
        q_peak = maximum(abs.(q), dims=2)
        min_val = minimum(q_peak)
        max_val = maximum(q_peak)
        q_peak_norm = (q_peak .- min_val) ./ (max_val - min_val)
        edge_vals = q_peak_norm
        cbar_title = "Normalised peak flow"
        # colorbar_ticks = (0:0.2:1)

    elseif flow_metric == "average"
        # average flow
        q_ave = mean(abs.(q), dims=2)
        min_val = minimum(q_ave)
        max_val = maximum(q_ave)
        q_ave_norm = (q_ave .- min_val) ./ (max_val - min_val)
        edge_vals = q_ave_norm
        cbar_title = "Normalised average flow"
        # colorbar_ticks = (0:0.2:1)

    elseif flow_metric == "flow_rev"
        # flow reversal count
        rev_count = zeros(network.np)
        
        for j in collect(1:network.np)
            a = 0
            for k in collect(2:network.nt)
                if sign(q[j, k - 1]) * sign(q[j, k]) == -1
                    b = 1
                else
                    b = 0
                end
                a += b
            end
            rev_count[j] = a
        end
        edge_vals = rev_count
        cbar_title = "Flow reversal count"
        colorbar_ticks = (0:2:10, [string.(round.(Int, (0:2:8))); L"$\geq 10$"])
        clims = (0, 10)
    end


    # edge_colour = cgrad(ColorSchemes.OrRd)
    # edge_colour = cgrad(ColorSchemes.PuBu)
    # edge_colour = cgrad(ColorSchemes.Reds[2:2:end-1])
    # edge_colour = cgrad(ColorSchemes.Greens[2:2:end-1])
    # edge_colour = cgrad(ColorSchemes.Blues[2:end])
    # edge_colour = cgrad(ColorSchemes.Reds)
    # edge_colour = cgrad(ColorSchemes.Blues)
    edge_colour = cgrad(ColorSchemes.Greens)
    # edge_colour = reverse(cgrad(ColorSchemes.RdYlBu))
    # edge_colour = cgrad([:green, :orange, :red], [0.1, 0.5, 0.9])

    line_colour = ColorSchemes.Reds[6]

    pos_x, pos_y = get_graphing_x_y(network)

    fig = Plots.plot()  

    # plot edges
    edge_x = []
    edge_y = []
    for (idx, edge) in enumerate(edges(G))
        push!(edge_x, pos_x[src(edge)])
        push!(edge_x, pos_x[dst(edge)])
        push!(edge_x, NaN)
        push!(edge_y, pos_y[src(edge)])
        push!(edge_y, pos_y[dst(edge)])
        push!(edge_y, NaN)
    end
    # fig = Plots.plot(edge_x, edge_y, mode="lines", frame_style=:none, linecolor=line_colour, linewidth=0.5, label="Observable path")

    # # plot observable paths
    # for (idx, edge) in enumerate(edges(G))
    #     fig = Plots.plot!([pos_x[src(edge)], pos_x[dst(edge)]], [pos_y[src(edge)], pos_y[dst(edge)]], linecolor=:grey85, linewidth=1, label="", linealpha=1)
    # end

    
    # if flow_metric == "flow_rev"

    #     for i in collect(1:network.np)
    #         i_dst = findall(A[i, :] .> 0)[1]
    #         i_src = findall(A[i, :] .< 0)[1]
    #         if i_dst in all_nodes && i_src in all_nodes && edge_vals[i] > 0
    #             fig = Plots.plot!([pos_x[i_src], pos_x[i_dst]], [pos_y[i_src], pos_y[i_dst]], mode="lines", line_z=edge_vals[i], linecolor=edge_colour, linewidth=2.5, label="", linealpha=1, colorbar_title=cbar_title, colorbar_ticks=colorbar_ticks)
    #         end
    #     end

    # else

    #     for i in collect(1:network.np)
    #         i_dst = findall(A[i, :] .> 0)[1]
    #         i_src = findall(A[i, :] .< 0)[1]
    #         if i_dst in all_nodes && i_src in all_nodes
    #             fig = Plots.plot!([pos_x[i_src], pos_x[i_dst]], [pos_y[i_src], pos_y[i_dst]], mode="lines", line_z=edge_vals[i], linecolor=edge_colour, linewidth=2.5, label="", linealpha=1, colorbar_title=cbar_title, colorbar_ticks=colorbar_ticks)
    #         end
    #     end

    # end

    # for i in collect(1:network.np)
    #     i_dst = findall(A[i, :] .> 0)[1]
    #     i_src = findall(A[i, :] .< 0)[1]
    #     if i_dst in all_nodes && i_src in all_nodes
    #         fig = Plots.plot!([pos_x[i_src], pos_x[i_dst]], [pos_y[i_src], pos_y[i_dst]], mode="lines", line_z=edge_vals[i], linecolor=edge_colour, linewidth=2.5, label="", linealpha=1, colorbar_title=cbar_title, colorbar_ticks=colorbar_ticks, clims=clims)
    #     end
    # end

    for i in collect(1:network.np)
        i_dst = findall(A[i, :] .> 0)[1]
        i_src = findall(A[i, :] .< 0)[1]
        fig = Plots.plot!([pos_x[i_src], pos_x[i_dst]], [pos_y[i_src], pos_y[i_dst]], mode="lines", line_z=edge_vals[i], linecolor=edge_colour, linewidth=2.5, label="", linealpha=1, colorbar_title=cbar_title, colorbar_ticks=colorbar_ticks, clims=clims)
    end

    # colorbar_ticks=colorbar_ticks, clims=clims

    # # plot edges
    # for (idx, edge) in enumerate(edges(G))

    #     if !(src(edge) in all_nodes) && !(dst(edge) in all_nodes)
    #         fig = Plots.plot([pos_x[src(edge)], pos_x[dst(edge)]], [pos_y[src(edge)], pos_y[dst(edge)]], frame_style=:none, linecolor=:grey70, linewidth=1.25, label="")
    #     end
    # end

    # plot sensor nodes
    sen_x = []
    sen_y = []
    for sen in sensor_idx
        push!(sen_x, pos_x[sen])
        push!(sen_y, pos_y[sen])
    end
    fig = Plots.scatter!(sen_x, sen_y, markershape=:circle, c=:black, markerstrokecolor=:white, markerstrokewidth=1, markeralpha=1, markersize=8, label="")

    fig = Plots.plot!(legendfont=12, titlefont=14, colorbar_tickfontsize=12, colorbar_titlefontsize=14, frame_style=:none, showaxis=false, grid=false, aspect_ratio=:equal)


end



# plot link flow data as time series
begin

    link_to_plot = "link_2747"
    # link_to_plot = "link_2748"

    x = df_flow[:, :timestamp]
    y = df_flow[:, link_to_plot]

    link_flow = Plots.plot()
    link_flow = Plots.plot!(x, y, label="")
    link_flow= Plots.plot!(xlabel="Simulation time [h]", ylabel="Flow [L/s]", xtickfontsize=16, ytickfontsize=16, xguidefontsize=18, yguidefontsize=18, legendfont=14, legend=:topright)
end


# plot node pressure data as time series
begin

    node_to_plot = "node_2197"

    x = df_pressure[:, :timestamp]
    y = df_pressure[:, node_to_plot]

    node_pressure = Plots.plot()
    link_pressure = Plots.plot!(x, y, label="")
    link_pressure = Plots.plot!(xlabel="Simulation time [h]", ylabel="Pressure [m]", xtickfontsize=16, ytickfontsize=16, xguidefontsize=18, yguidefontsize=18, legendfont=14, legend=:topright)
end


# plot node quality data as time series
begin

    # node_to_plot = "node_0187"
    # node_to_plot = "node_2367"
    # node_to_plot = "node_1802"
    # node_to_plot = "node_1773"
    node_to_plot = "node_1809"
    # node_to_plot = "node_2506"
    # node_to_plot = "node_1811"

    x = df_qual[:, :timestamp]
    y = df_qual[:, node_to_plot]

    node_pressure = Plots.plot()
    link_pressure = Plots.plot!(x, y, label="")
    link_pressure = Plots.plot!(xlabel="Simulation time [h]", ylabel="Chlorine [mg/L]", xtickfontsize=16, ytickfontsize=16, xguidefontsize=18, yguidefontsize=18, legendfont=14, legend=:topright)
end


