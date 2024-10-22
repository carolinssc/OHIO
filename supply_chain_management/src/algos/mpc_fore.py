import os, sys
import subprocess
from collections import defaultdict
import codecs

sys.path.append(os.getcwd())
"""
implements the MPC baseline
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

def solveMCP(
    env,
    CPLEXPATH=None,
    res_path="scim",
    directory="saved_files",
    T=30,
):
  
    t = env.period

    factory_nodes = env.factory
    distribution_nodes = env.distrib
    retail_nodes = env.retail
    
    nodes = factory_nodes + distribution_nodes
   
    nodes.sort()
    edges = env.reorder_links

    s = defaultdict(lambda: 0)
    for n in distribution_nodes + factory_nodes:
        s[n] = env.X[n].iloc[env.period - 1]

    u = defaultdict(lambda: 0)
    h = defaultdict(lambda: 0)
    for n in nodes:
        if "h" in env.graph.nodes[n]:
            h[n] = env.graph.nodes[n]["h"]
        if "o" in env.graph.nodes[n]:
            u[n] = env.graph.nodes[n]["o"]

    demand = defaultdict(lambda: 0)
    b = defaultdict(lambda: 0)
    g = defaultdict(lambda: 0)
    L = defaultdict(lambda: 0)
    p = defaultdict(lambda: 0)
    demand = []
    
    for i, j in env.retail_links:
        demand.append((i, 1, env.D.loc[env.period, (i, j)]))
        for tt in range(t + 1, t + T):
            if tt < env.num_periods:
               
                demand.append((i, tt - t + 1, env.demand_rate[(i, j)][tt]))
            else:
                demand.append((i, tt - t + 1, 0))
        b[i, j] = env.graph.edges[(i, j)]["b"]
        p[i, j] = env.graph.edges[(i, j)]["p"]

    arrival = []
    for i, j in env.reorder_links:
        g[i, j] = env.graph.edges[(i, j)]["g"]
        L[i, j] = env.graph.edges[(i, j)]["L"]
        p[i, j] = env.graph.edges[(i, j)]["p"]
        
        for tt in range(t, t + T):
            if tt - L[i, j] >= 1 and tt - L[i, j] < env.num_periods:
                arrival.append((i, j, tt - t + 1, env.R.loc[tt - L[i, j], (i, j)]))
            else:
                arrival.append((i, j, tt - t + 1, 0))
  
    unmetDemand = defaultdict(lambda: 0)
    for i, j in env.retail_links:
        unmetDemand[i, j] = env.U.loc[env.period - 1, (i, j)]


    prod = defaultdict(lambda: 0)
    for l in env.factory:
        prod[l] = env.Prod.loc[t-1,l]
  
    edgeAttr = [
        (
            i,
            j,
            g[i, j],
            L[i, j],
        )
        for (i, j) in edges
    ]
 
    C = defaultdict(lambda: 0)
    for i in factory_nodes:
        C[i] = env.graph.nodes[(i)]["C"]

    nodeAttr = [
        (
            i,
            C[i],
            u[i],
            h[i],
            s[i],
            env.graph.nodes[i]["Cap"],
            p[i,j], 
            unmetDemand[i, j],
            b[i, j],
            prod[i],
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
        file.write("T=" + str(T) + ";\r\n")
        file.write('path="' + resfile + '";\r\n')
        file.write("factory_nodes=" + mat2str(factory_nodes) + ";\r\n")
        file.write("distribution_nodes=" + mat2str(distribution_nodes) + ";\r\n")
        file.write("retail_nodes=" + mat2str(retail_nodes) + ";\r\n")
        file.write("nodeAttr=" + mat2str(nodeAttr) + ";\r\n")
        file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
        file.write("d_r_nodes=" + mat2str(distribution_nodes) + ";\r\n")
        file.write("reorder_e=" + mat2str(env.reorder_links) + ";\r\n")
        file.write("demand=" + mat2str(demand) + ";\r\n")
        file.write("arrival=" + mat2str(arrival) + ";\r\n")
  
    modfile = modPath + "mpc_fore.mod"
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
    SR = defaultdict(float)
    PC = defaultdict(float)
    TC = defaultdict(float)
    OC = defaultdict(float)
    HC = defaultdict(float)
    UP = defaultdict(float)
    UD = defaultdict(float)
    production = defaultdict(float)
    inventory = defaultdict(float)
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
                .split("=")
            )

            if item[0] == "SR":
                vaules = item[1].strip(")]").strip("[(").split(")(")
                for v in vaules:
                    if len(v) == 0:
                        continue
                    t, V = v.split(",")
                    SR[t] = float(V)
            if item[0] == "PC":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    t, V = v.split(",")
                    PC[t] = float(V)
            if item[0] == "TC":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    t, V = v.split(",")
                    TC[t] = float(V)
            if item[0] == "OC":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    t, V = v.split(",")
                    OC[t] = float(V)
            if item[0] == "HC":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    t, V = v.split(",")
                    HC[t] = float(V)
            if item[0] == "UP":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    t, V = v.split(",")
                    UP[t] = float(V)
            if item[0] == "unfulfilled_demand":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, t, V = v.split(",")
                    UD[t] = float(V)
            if item[0] == "inventory":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, t, V = v.split(",")
                    inventory[int(i), int(t)] = float(V)
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
            if item[0] == "production":
                values = item[1].strip(")]").strip("[(").split(")(")

                for v in values:
                    if len(v) == 0:
                        continue
                
                    i, P = v.split(",")
                    production[int(i)] = float(P)

                    """
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
                    """

    results_MPC = {
        "SR": SR,
        "PC": PC,
        "TC": TC,
        "OC": OC,
        "HC": HC,
        "UP": UP,
        "UD": UD,
        "inventory": inventory,
        "production": production,
    }
    return flow, production, results_MPC
