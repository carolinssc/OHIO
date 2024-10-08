import os, sys
import subprocess
from collections import defaultdict
import codecs

sys.path.append(os.getcwd())

"""
This module contains the function solveLCP, which is used to solve the lower-level optimization for the supply chain model.
"""

def mat2str(mat):
    return (
        str(mat)
        .replace("'", '"')
        .replace("(", "<")
        .replace(")", ">")
        .replace("[", "{")
        .replace("]", "}")
    )

def solveLCP(
    env,
    desiredDistrib=None,
    desiredProd=None,
    CPLEXPATH=None,
    res_path="scim",
    directory="saved_files",
):
    t = env.period

    factory_nodes = env.factory
    distribution_nodes = env.distrib


    nodes = sorted(factory_nodes + distribution_nodes)
    edges = [(i, j) for i, j in env.graph.edges if i in nodes and j in nodes]
   
    availInv = [env.X[n].iloc[env.period - 1] for n in factory_nodes] + [
        env.Prod[n].iloc[env.period - 1] for n in factory_nodes
    ]
    
    desiredInv = {}
    for i in nodes:
        desiredInv[i] = int(desiredDistrib[i - 1] * sum([v for v in availInv]))
   

    desiredInv = [(i, desiredInv[i]) for i in nodes]

    desiredProd = [(i, int(desiredProd)) for i in factory_nodes]
    
    demand = defaultdict(lambda: 0)  # fulfilled demand
    for i, j in env.retail_links:
        
        if t < env.num_periods - 1:
            demand[i] = (
                env.D.loc[t, (i, j)]
                + env.U.loc[t - 1, (i, j)]
                + env.demand_rate[(i, j)][t + 1]
            )
        else:
            demand[i] = env.D.loc[t, (i, j)] + env.U.loc[t - 1, (i, j)]

    arrival = []
    a = defaultdict(lambda: 0)
    for i, j in env.reorder_links:
        for tt in range(t, t + env.lt_max + 1):
            a[j] += env.Y.loc[tt, (i, j)]
   
    for l in factory_nodes:
        a[l] = env.Prod.loc[t - 1, l]
    for i in nodes:
        arrival.append((i, int(a[i])))
  
    edgeAttr = [
        (
            i,
            j,
            env.graph.edges[(i, j)]["g"],
            env.graph.edges[(i, j)]["L"],
            env.graph.edges[(i, j)]["p"],
        )
        for (i, j) in edges
    ]
    C = defaultdict(lambda: 0)
    for i in factory_nodes:
        C[i] = env.graph.nodes[(i)]["C"]

    nodeAttr = [
        (
            i,
            env.X[i].iloc[env.period - 1],  # current inventory
            demand[i],  # demand
            env.graph.nodes[i]["Cap"],  # storage capacity
            C[i],  # production capacity
        )
        for i in nodes
    ]
    

    modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
    matchingPath = (
        os.getcwd().replace("\\", "/")
        + "/"
        + directory
        + "/cplex_logs/"
        + res_path
        + "/"
    )

    if not os.path.exists(matchingPath):
        os.makedirs(matchingPath)
    datafile = matchingPath + "data_{}.dat".format(t)
    resfile = matchingPath + "res_{}.dat".format(t)
    with open(datafile, "w") as file:
        file.write('path="' + resfile + '";\r\n')
        file.write("desiredInv=" + mat2str(desiredInv) + ";\r\n")
        file.write("desiredProd=" + mat2str(desiredProd) + ";\r\n")
        file.write("factory_nodes=" + mat2str(factory_nodes) + ";\r\n")
        file.write("distribution_nodes=" + mat2str(distribution_nodes) + ";\r\n")
        file.write("nodeAttr=" + mat2str(nodeAttr) + ";\r\n")
        file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
        file.write(
            "f_d_r_nodes=" + mat2str(factory_nodes + distribution_nodes) + ";\r\n"
        )
        file.write("arrival=" + mat2str(arrival) + ";\r\n")

        modfile = modPath + "lcp_cap.mod"
    if CPLEXPATH is None:
        CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file = matchingPath + "out_{}.dat".format(t)
    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )
    output_f.close()
    flow = defaultdict(float)
    error = defaultdict(float)
    prod = defaultdict(float)
    with codecs.open(resfile, "r", encoding="utf8", errors="ignore") as file:
        for row in file:
            item = (
                row.replace("e)", ")")
                .strip()
                .strip(";")
                .replace("?", "")
                .replace("\x1f", "")
                .replace("\x0f", "")
                .replace("\x7f", "")
                .replace("/", "")
                .replace("O", "")
                .split("=")
            )
            if item[0] == "flow":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, f = v.split(",")
                    try:
                        flow[int(i), int(j)] = float(f)
                    except:
                        print(item)
                        print(v)
                        print(f)
                    flow[int(i), int(j)] = float(
                        f.replace("y6", "")
                        .replace("I0", "")
                        .replace("\x032", "")
                        .replace("C8", "")
                        .replace("C3", "")
                        .replace("c5", "")
                        .replace("#9", "")
                        .replace("c9", "")
                        .replace("\x132", "")
                        .replace("c2", "")
                        .replace("\x138", "")
                        .replace("c2", "")
                        .replace("\x133", "")
                        .replace("\x131", "")
                        .replace("s", "")
                        .replace("#0", "")
                        .replace("c4", "")
                        .replace("\x031", "")
                        .replace("c8", "")
                        .replace("\x037", "")
                        .replace("\x034", "")
                        .replace("s4", "")
                        .replace("S3", "")
                        .replace("\x139", "")
                        .replace("\x138", "")
                        .replace("C4", "")
                        .replace("\x039", "")
                        .replace("S8", "")
                        .replace("\x033", "")
                        .replace("S5", "")
                        .replace("#", "")
                        .replace("\x131", "")
                        .replace("\t6", "")
                        .replace("\x01", "")
                        .replace("i9", "")
                        .replace("y4", "")
                        .replace("a6", "")
                        .replace("y5", "")
                        .replace("\x018", "")
                        .replace("I5", "")
                        .replace("\x11", "")
                        .replace("y2", "")
                        .replace("\x011", "")
                        .replace("y4", "")
                        .replace("y5", "")
                        .replace("a2", "")
                        .replace("i9", "")
                        .replace("i7", "")
                        .replace("\t3", "")
                        .replace("q", "")
                        .replace("I3", "")
                        .replace("A", "")
                        .replace("y5", "")
                        .replace("Q", "")
                        .replace("a3", "")
                        .replace("\x190", "")
                        .replace("\x013", "")
                        .replace("o", "")
                        .replace("`", "")
                        .replace("\x10", "")
                        .replace("P", "")
                        .replace("p", "")
                        .replace("@", "")
                        .replace("M", "")
                        .replace("]", "")
                        .replace("?", "")
                        .replace("\x1f", "")
                        .replace("}", "")
                        .replace("m", "")
                        .replace("\x04", "")
                        .replace("\x0f", "")
                        .replace("\x7f", "")
                        .replace("T", "")
                        .replace("$", "")
                        .replace("t", "")
                        .replace("\x147", "")
                        .replace("\x14", "")
                        .replace("\x046", "")
                        .replace("\x042", "")
                        .replace("/", "")
                        .replace("O", "")
                        .replace("D", "")
                        .replace("d", "")
                        .replace(")", "")
                        .replace("Y", "")
                        .replace("i", "")
                        .replace("\x193", "")
                        .replace("\x192", "")
                        .replace("y5", "")
                        .replace("I2", "")
                        .replace("\t", "")
                        .replace("i2", "")
                        .replace("!", "")
                        .replace("i7", "")
                        .replace("A8", "")
                    )
            if item[0] == "error":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, p = v.split(",")
                    error[int(i)] = float(p)

            if item[0] == "production":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, p = v.split(",")
                    prod[int(i)] = float(p)

    ship = {(i, j): flow[i, j] if (i, j) in flow else 0 for i, j in edges}
    error = [error[i] for i in distribution_nodes + factory_nodes]
    prod = {(i): prod[i] for i in factory_nodes}
    action = ship
    return action, prod, error
