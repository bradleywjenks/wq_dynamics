using Revise
using OpWater

### load network data ###
begin

    net_name = "bwfl_2022_05_hw"

    # load network data
    if net_name == "bwfl_2022_05_hw"
        load_name = "bwfl_2022_05/hw"
    else
        load_name = net_name
    end
    network = load_network(load_name, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=true)

    # assign control valve locations
    # inputs from optimisation results
    prv_new_idx = [1963, 749, 1778]
    afv_new_idx = [1285, 1236, 2444, 2095]

    # inputs from existing controls
    prv_exist_idx = [2755, 2728, 2742]
    dbv_idx = [2746, 2747]

    bv_idx = [2388, 2430, 2554, 2507, 2235]

end

### plotting code ###

# separated DMAs
network.pcv_loc = prv_exist_idx
network.bv_loc = vcat(dbv_idx, bv_idx[2])
plot_network_layout(network, pipes=true, reservoirs=true, pcvs=true, bvs=true, dbvs=false, afvs=false, legend=true)

# existing controls
network.pcv_loc = prv_exist_idx
network.dbv_loc = dbv_idx
network.bv_loc = [bv_idx[2]]
plot_network_layout(network, pipes=true, reservoirs=true, pcvs=true, bvs=true, dbvs=true, afvs=false, legend=true)

# scc controls
network.pcv_loc = prv_new_idx
network.afv_loc = afv_new_idx
plot_network_layout(network, pipes=true, reservoirs=true, pcvs=true, bvs=false, dbvs=false, afvs=true, legend=true)