import os
import subprocess
from collections import defaultdict
from src.misc.utils import mat2str
import numpy as np
from gurobipy import Model, GRB, quicksum, Env

"""
This file contains the lower-level optimization, as well as the DTV and INF algorithms.
"""


def solveRebFlow(env, res_path, desiredAcc, CPLEXPATH, directory, minobj=0):
    t = env.time
    accRLTuple = [(n, int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
    edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]
    modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
    # OPTPath = os.getcwd().replace('\\','/')+'/' + directory +'/cplex_logs/rebalancing/'+res_path + '/'
    OPTPath = directory + "/cplex_logs/rebalancing/" + res_path + "/"
    if not os.path.exists(OPTPath):
        os.makedirs(OPTPath)
    datafile = OPTPath + f"data_{t}.dat"
    resfile = OPTPath + f"res_{t}.dat"

    with open(datafile, "w") as file:
        file.write('path="' + resfile + '";\r\n')
        file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
        file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
        file.write("accRLTuple=" + mat2str(accRLTuple) + ";\r\n")
        file.write("minobj=" + str(minobj) + ";\r\n")
    modfile = modPath + "minRebDistRebOnly.mod"
    if CPLEXPATH is None:
        CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file = OPTPath + f"out_{t}.dat"
    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )
    output_f.close()
    obj = None
    # 3. collect results from file
    flow = defaultdict(float)
    with open(resfile, "r", encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(";").split("=")
            if item[0] == "flow":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, f = v.split(",")
                    flow[int(i), int(j)] = float(f)
            if item[0] == "obj":
                obj = float(item[1])
    action = [flow[i, j] for i, j in env.edges]
    return action


def DTV(env, demand):
    # Create the model
    gb_env = Env(empty=True)
    gb_env.setParam("OutputFlag", 0)
    gb_env.start()
    model = Model("mincostflow", env=gb_env)

    t = env.time

    accInitTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
    edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]

    region = [i for (i, n) in accInitTuple]
    # add self loops to edgeAttr with time =0
    edgeAttr += [(i, i, 0) for i in region]
    time = {(i, j): t for (i, j, t) in edgeAttr}

    vehicles = {i: v for (i, v) in accInitTuple}

    num_vehicles = sum(vehicles.values())
    num_requests = sum(demand.values())

    vehicle_region = {}
    vehicle = 0
    for i in region:
        for _ in range(vehicles[i]):
            vehicle_region[vehicle] = i
            vehicle += 1

    request_region = {}
    request = 0
    for i in demand.keys():
        for _ in range(int(demand[i])):
            request_region[request] = i
            request += 1

    # calculate time for each vehicle to each request according to the region
    time_vehicle_request = {}
    for vehicle in vehicle_region:
        for request in request_region:
            time_vehicle_request[vehicle, request] = time[
                vehicle_region[vehicle], request_region[request]
            ]

    edge = [
        (vehicle, request) for vehicle in vehicle_region for request in request_region
    ]

    rebFlow = model.addVars(edge, vtype=GRB.BINARY, name="x")
    model.update()
    # Set objective
    model.setObjective(
        quicksum(rebFlow[e] * time_vehicle_request[e] for e in edge), GRB.MINIMIZE
    )

    # Add constraints
    model.addConstr(
        quicksum(rebFlow[v, k] for v, k in edge) == min(num_vehicles, num_requests)
    )

    # only one vehicle can be assigned to one request
    for request in request_region:
        model.addConstr(quicksum(rebFlow[v, request] for v in vehicle_region) <= 1)

    # only one request can be assigned to one vehicle
    for vehicle in vehicle_region:
        model.addConstr(quicksum(rebFlow[vehicle, k] for k in request_region) <= 1)

    # Optimize the model
    model.optimize()

    # get rebalancing flows
    flows = {e: 0 for e in env.edges}
    for var in model.getVars():
        if var.X != 0:
            substring = var.VarName[
                var.VarName.index("[") + 1 : var.VarName.index("]")
            ].split(",")
            i = vehicle_region[int(substring[0])]
            j = request_region[int(substring[1])]
            flows[i, j] += 1

    action = [flows[i, j] for i, j in env.edges]
    return action, flows


def INF(env, demand_rate, max_reb, roh):
    # Create the model
    gb_env = Env(empty=True)
    gb_env.setParam("OutputFlag", 0)
    gb_env.start()
    model = Model("LP2", env=gb_env)

    t = env.time

    accInitTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
    edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]

    region = [i for (i, n) in accInitTuple]

    # add self loops to edgeAttr with time =0
    edgeAttr += [(i, i, 0) for i in region]

    time = {(i, j): t for (i, j, t) in edgeAttr}

    vehicles = {i: v for (i, v) in accInitTuple}

    vehicle_region = {}
    vehicle = 0
    for i in region:
        for _ in range(vehicles[i]):
            vehicle_region[vehicle] = i
            vehicle += 1

    # calculate time for each vehicle to each region
    time_vehicle_region = {}
    for vehicle in vehicle_region:
        for r in region:
            time_vehicle_region[vehicle, r] = time[vehicle_region[vehicle], r]

    edge = [(vehicle, r) for vehicle in vehicle_region for r in region]

    rebFlow = model.addVars(edge, vtype=GRB.BINARY, name="x")
    model.update()
    # Set objective
    model.setObjective(
        quicksum(
            rebFlow[e] * demand_rate[e[1]] * (max_reb - time_vehicle_region[e])
            for e in edge
        ),
        GRB.MAXIMIZE,
    )
    # Add constraints
    # only one vehicle can be assigned to one region
    for v in vehicle_region:
        model.addConstr(quicksum(rebFlow[v, k] for k in region) <= 1)
        for j in region:
            model.addConstr(rebFlow[v, j] * (max_reb - time_vehicle_region[v, j]) >= 0)
    for i in region:
        model.addConstr(
            quicksum(
                rebFlow[v, i] * (max_reb - time_vehicle_region[i, j])
                for v in vehicle_region
            )
            <= demand_rate[i] * roh * max_reb**2
        )

    # Optimize the model
    model.optimize()

    # get rebalancing flows
    flows = {e: 0 for e in env.G.edges}
    for i in region:
        flows[(i, i)] = 0
    for var in model.getVars():
        if var.X != 0:
            substring = var.VarName[
                var.VarName.index("[") + 1 : var.VarName.index("]")
            ].split(",")
            i = vehicle_region[int(substring[0])]
            j = int(substring[1])
            flows[i, j] += 1

    action = [flows[i, j] for i, j in env.edges]
    return action, flows
