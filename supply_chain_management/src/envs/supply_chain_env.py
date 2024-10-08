import gymnasium as gym
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import *
from torch_geometric.data import Data
import torch
import random

"This file is build on top of: https://github.com/hubbs5/or-gym"

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=12, edge_list=None):
        super().__init__()
        self.env = env
        self.T = T
        self.edge_list = edge_list
        self.edge_list_bidirectional = edge_list + [(j, i) for (i, j) in edge_list]
        print("T", T)

    def parse_obs(self, pipeline):
        x = (
            torch.cat(
                (
                    torch.cat(
                        (
                            torch.tensor(
                                [
                                    self.env.D[d].iloc[self.env.period]
                                    + self.env.U[d].iloc[self.env.period - 1]
                                    for d in self.env.retail_links
                                ]
                            ).view(1, len(self.env.retail_links), 1),
                            torch.zeros(
                                1, len(self.env.nodes) - len(self.env.retail), 1
                            ),
                        ),
                        dim=1,
                    ),
                    # holding cost at each node
                    torch.tensor(
                        [
                            self.env.graph.nodes[n]["h"]
                            if "h" in self.env.graph.nodes[n]
                            else 0
                            for n in self.env.nodes
                        ]
                    ).view(1, len(self.env.nodes), 1),
                    # operating cost at each node
                    torch.tensor(
                        [
                            self.env.graph.nodes[n]["o"]
                            if "o" in self.env.graph.nodes[n]
                            else 0
                            for n in self.env.nodes
                        ]
                    ).view(1, len(self.env.nodes), 1),
                    # incoming product flows
                    torch.tensor([pipeline.T]).view(1, len(self.env.nodes), self.T),
                    torch.tensor(
                        [
                            self.env.X[n].iloc[self.env.period - 1]
                            for n in self.env.nodes
                        ]
                    ).view(1, len(self.env.nodes), 1),
                    torch.cat(
                        (
                            torch.tensor(
                                [
                                    [
                                        self.env.demand_rate[i, j][self.env.period + tt]
                                        if self.env.period + tt < self.env.num_periods
                                        else 0
                                        for tt in range(1, 11)
                                    ]
                                    for i, j in self.env.retail_links
                                ]
                            ).view(1, len(self.env.retail_links), 10),
                            torch.zeros(
                                1, len(self.env.nodes) - len(self.env.retail), 10
                            ),
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            torch.tensor(
                                [
                                    self.env.graph.edges[i, j]["p"]
                                    for (i, j) in self.env.retail_links
                                ]
                            ).view(1, len(self.env.retail_links), 1),
                            torch.zeros(
                                1, len(self.env.nodes) - len(self.env.retail), 1
                            ),
                        ),
                        dim=1,
                    ),
                    torch.tensor(
                        [self.env.graph.nodes[(i)]["Cap"] for i in self.env.nodes]
                    ).view(1, len(self.env.nodes), 1),
                    torch.cat(
                        (
                            torch.zeros(
                                1, len(self.env.nodes) - len(self.env.factory), 1
                            ),
                            torch.tensor(
                                [
                                    self.env.graph.nodes[(i)]["C"]
                                    for i in self.env.factory
                                ]
                            ).view(1, len(self.env.factory), 1),
                        ),
                        dim=1,
                    ),
                ),
                dim=-1,
            )
            .view(len(self.env.nodes), 18)
            .float()
        )

        # Edge features
        edge_index = torch.tensor(self.edge_list_bidirectional).T.long()

        # torch.cat(
        # (
        e = (
            torch.tensor(
                [
                    self.env.graph.edges[i + 1, j + 1]["L"]
                    for (i, j) in self.edge_list * 2
                ]
            )
            .view(1, edge_index.shape[1])
            .float()
            .T
        )
        # torch.tensor(
        #    [
        #        self.env.graph.edges[i + 1, j + 1]["g"]
        #        for (i, j) in self.edge_list * 2
        #    ]
        # )
        # .view(1, edge_index.shape[1])
        # .float(),
        # ),
        # dim=0,
        # )
        # .view(2, edge_index.shape[1])
        # .T

        data = Data(x, edge_index, edge_attr=e)
        return data


def get_demand_curve(dmax=25, dvar=2, T=30, r=1, f=4):
    demand = []
    demand_rate = []
    for t in range(T):
        demand.append(
            max(
                0,
                np.round(
                    dmax / 2
                    + dmax / 2 * np.cos(f * np.pi * (2 * r + t) / T)
                    + np.random.randint(-(dvar / 2 + 1), dvar / 2 + 1)
                ),
            )
        )
        demand_rate.append(
            np.round(dmax / 2 + dmax / 2 * np.cos(f * np.pi * (2 * r + t) / T))
        )
    return demand, demand_rate

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, "env_config"):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self, key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key, type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")

class NetInvMgmtMasterEnv(gym.Env):

    def __init__(self, version=1, **kwargs):
        """
        num_periods = number of periods in simulation.
        Node specific parameters:
            - I0 = initial inventory.
            - C = production capacity.
            - o = unit operating cost (feed-based)
            - h = unit holding cost for excess on-hand inventory.
            - Cap = maximum inventory capacity.
        Edge specific parameters:
            - L = lead times in betwen adjacent nodes.
            - p = unit price to send material between adjacent nodes (purchase price/reorder cost)
            - b = unit backlog cost or good-wil loss for unfulfilled market demand between adjacent retailer and market.
            - g = unit holding cost for pipeline inventory on a specified edge.
          
        backlog = Are unfulfilled orders backlogged? True = backlogged, False = lost sales.
        alpha = discount factor in the range (0,1] that accounts for the time value of money
        seed_int = integer seed for random state.
        user_D = dictionary containing user specified demand (list) for each (retail, market) pair at
            each time period in the simulation. If all zeros, ignored; otherwise, demands will be taken from this list.
        sample_path = dictionary specifying if is user_D (for each (retail, market) pair) is sampled from demand_dist.
        """
        # set default (arbitrary) values when creating environment (if no args or kwargs are given)
        print("version", version)
        self.version = version
     
        self.num_periods = 30 + 1
        self.backlog = True
        self.alpha = 1.00
        self.seed_int = 0
        self.user_D = {}
        self.demand_rate = {}

        self.sample_path = {(1, 0): False}

        self.use_mlp = False
        # create graph
        self.graph = nx.DiGraph()
        # Market
        self.graph.add_nodes_from([0])

        if self.version == 1:
            # toy example,two nodes
            self.user_D[(1, 0)], self.demand_rate[(1, 0)] = get_demand_curve(
                dmax=25, T=self.num_periods
            )
            # create graph
            self.graph = nx.DiGraph()
            # Market
            self.graph.add_nodes_from([0]),
            # Retailer
            self.graph.add_nodes_from([1], I0=10, h=1, Cap=30)  # 100, 0.03
            # Manufacturers
            self.graph.add_nodes_from([2], I0=0, C=20, o=0.80, v=1.000, h=0.001, Cap=50)
            # Links
            self.graph.add_edges_from(
                [
                    (1, 0, {"p": 7.000, "b": 1.500}),
                    (2, 1, {"L": 2, "p": 1.500, "g": 1}),  # 0.01
                ]
            )

        if self.version == 2:
            # 1F3S
            dmax = [5, 15, 20]
            r = [1, 3, 6]
            f = [2, 4, 6]

            self.user_D[(1, 0)], self.demand_rate[(1, 0)] = get_demand_curve(
                dmax=dmax[0],
                T=self.num_periods,
                r=r[0],
                f=f[0],
                dvar=2,
            )

            self.user_D[(2, 0)], self.demand_rate[(2, 0)] = get_demand_curve(
                dmax=dmax[1],
                T=self.num_periods,
                r=r[1],
                f=f[1],
                dvar=2,
            )

            self.user_D[(3, 0)], self.demand_rate[(3, 0)] = get_demand_curve(
                dmax=dmax[2],
                T=self.num_periods,
                r=r[2],
                f=f[2],
                dvar=2,
            )

            # Retailer
            self.graph.add_nodes_from([1], I0=10, h=0.5, Cap=15)
            self.graph.add_nodes_from([2], I0=10, h=0.5, Cap=15)
            self.graph.add_nodes_from([3], I0=10, h=0.5, Cap=15)
            # Manufacturers
            self.graph.add_nodes_from([4], I0=0, C=25, o=5, v=1.000, h=0.1, Cap=50)
            # Links
            self.graph.add_edges_from(
                [
                    (1, 0, {"p": 15.000, "b": 1.00}),
                    (2, 0, {"p": 15.000, "b": 1.00}),
                    (3, 0, {"p": 15.00, "b": 1.00}),
                    (4, 1, {"L": 1, "p": 6.000, "g": 0.5}),
                    (4, 2, {"L": 1, "p": 6.000, "g": 0.5}),
                    (4, 3, {"L": 1, "p": 6.000, "g": 0.5}),
                ]
            )
        
        if self.version == 3:
            # 1F10S
            dmax = [5] * 4 + [10] * 3 + [18] * 3
            # dmax = [20, 25, 30, 20, 25, 30, 20, 25, 30, 20]
            r = [1, 1, 1, 3, 3, 3, 6, 6, 6, 2]
            f = [2, 4, 6, 2, 4, 6, 2, 4, 6, 3]

            self.user_D[(1, 0)], self.demand_rate[(1, 0)] = get_demand_curve(
                dmax=dmax[0], T=self.num_periods, r=r[0], f=f[0]
            )

            self.user_D[(2, 0)], self.demand_rate[(2, 0)] = get_demand_curve(
                dmax=dmax[1], T=self.num_periods, r=r[1], f=f[1]
            )

            self.user_D[(3, 0)], self.demand_rate[(3, 0)] = get_demand_curve(
                dmax=dmax[2], T=self.num_periods, r=r[2], f=f[2]
            )

            self.user_D[(4, 0)], self.demand_rate[(4, 0)] = get_demand_curve(
                dmax=dmax[3], T=self.num_periods, r=r[3], f=f[3]
            )
            self.user_D[(5, 0)], self.demand_rate[(5, 0)] = get_demand_curve(
                dmax=dmax[4], T=self.num_periods, r=r[4], f=f[4]
            )
            self.user_D[(6, 0)], self.demand_rate[(6, 0)] = get_demand_curve(
                dmax=dmax[5], T=self.num_periods, r=r[5], f=f[5]
            )
            self.user_D[(7, 0)], self.demand_rate[(7, 0)] = get_demand_curve(
                dmax=dmax[6], T=self.num_periods, r=r[6], f=f[6]
            )
            self.user_D[(8, 0)], self.demand_rate[(8, 0)] = get_demand_curve(
                dmax=dmax[7], T=self.num_periods, r=r[7], f=f[7]
            )
            self.user_D[(9, 0)], self.demand_rate[(9, 0)] = get_demand_curve(
                dmax=dmax[8], T=self.num_periods, r=r[8], f=f[8]
            )
            self.user_D[(10, 0)], self.demand_rate[(10, 0)] = get_demand_curve(
                dmax=dmax[9], T=self.num_periods, r=r[9], f=f[9]
            )

            # Retailer
            self.graph.add_nodes_from([1], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([2], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([3], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([4], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([5], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([6], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([7], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([8], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([9], I0=5, h=2, Cap=15)
            self.graph.add_nodes_from([10], I0=5, h=2, Cap=15)
            # Manufacturers
            self.graph.add_nodes_from([11], I0=0, C=60, o=5, v=1.000, h=0.005, Cap=80)
            # Links
            self.graph.add_edges_from(
                [
                    (1, 0, {"p": 15.000, "b": 1.500}),
                    (2, 0, {"p": 15.000, "b": 1.500}),
                    (3, 0, {"p": 15.000, "b": 1.500}),
                    (4, 0, {"p": 15.000, "b": 1.500}),
                    (5, 0, {"p": 15.000, "b": 1.500}),
                    (6, 0, {"p": 15.000, "b": 1.500}),
                    (7, 0, {"p": 15.000, "b": 1.500}),
                    (8, 0, {"p": 15.000, "b": 1.500}),
                    (9, 0, {"p": 15.000, "b": 1.500}),
                    (10, 0, {"p": 15.000, "b": 1.500}),
                    (11, 1, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 2, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 3, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 4, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 5, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 6, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 7, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 8, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 9, {"L": 1, "p": 6.000, "g": 0.5}),
                    (11, 10, {"L": 1, "p": 6.000, "g": 0.5}),
                ]
            )
        # add environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)

        # Save user_D and sample_path to graph metadata
        for link in self.user_D.keys():
            d = self.user_D[link]
            if np.sum(d) != 0:
                self.graph.edges[link]["user_D"] = d
                if link in self.sample_path.keys():
                    self.graph.edges[link]["sample_path"] = self.sample_path[link]
            else:
                # Placeholder to avoid key errors
                self.graph.edges[link]["user_D"] = 0

        self.num_nodes = self.graph.number_of_nodes()

        self.adjacency_matrix = np.vstack(self.graph.edges())
        # Set node levels
        self.levels = {}
        self.levels["retailer"] = np.array([1])
        self.levels["distributor"] = np.unique(
            np.hstack(
                [list(self.graph.predecessors(i)) for i in self.levels["retailer"]]
            )
        )
        self.levels["manufacturer"] = np.unique(
            np.hstack(
                [list(self.graph.predecessors(i)) for i in self.levels["distributor"]]
            )
        )

        self.level_col = {
            "retailer": 0,
            "distributor": 1,
            "manufacturer": 2,
            "raw_materials": 3,
        }

        self.market = [
            j for j in self.graph.nodes() if len(list(self.graph.successors(j))) == 0
        ]
        self.distrib = [
            j
            for j in self.graph.nodes()
            if "C" not in self.graph.nodes[j] and "I0" in self.graph.nodes[j]
        ]

        self.retail = [
            j
            for j in self.graph.nodes()
            if len(set.intersection(set(self.graph.successors(j)), set(self.market)))
            > 0
        ]
        self.factory = [j for j in self.graph.nodes() if "C" in self.graph.nodes[j]]

        self.main_nodes = np.sort(self.distrib + self.factory)
        self.nodes = np.sort(self.distrib + self.factory)
        self.main_edges = [
            (i, j)
            for i, j in self.graph.edges()
            if i in self.main_nodes and j in self.main_nodes
        ]
        self.reorder_links = [
            e for e in self.graph.edges() if "L" in self.graph.edges[e]
        ]  # exclude links to markets (these cannot have lead time 'L')
        self.retail_links = [
            e for e in self.graph.edges() if "L" not in self.graph.edges[e]
        ]  # links joining retailers to markets
        self.network_links = [
            e for e in self.graph.edges()
        ]  # all links involved in sale in the network
        print(self.levels)
        # check inputs
        assert set(self.graph.nodes()) == set.union(
            set(self.market), set(self.distrib), set(self.factory)
        ), "The union of market, distribution, factory, and raw material nodes is not equal to the system nodes."
        for j in self.graph.nodes():
            if "I0" in self.graph.nodes[j]:
                assert (
                    self.graph.nodes[j]["I0"] >= 0
                ), "The initial inventory cannot be negative for node {}.".format(j)
            if "h" in self.graph.nodes[j]:
                assert (
                    self.graph.nodes[j]["h"] >= 0
                ), "The inventory holding costs cannot be negative for node {}.".format(
                    j
                )
            if "C" in self.graph.nodes[j]:
                assert (
                    self.graph.nodes[j]["C"] > 0
                ), "The production capacity must be positive for node {}.".format(j)
            if "o" in self.graph.nodes[j]:
                assert (
                    self.graph.nodes[j]["o"] >= 0
                ), "The operating costs cannot be negative for node {}.".format(j)
            if "v" in self.graph.nodes[j]:
                assert (
                    self.graph.nodes[j]["v"] > 0 and self.graph.nodes[j]["v"] <= 1
                ), "The production yield must be in the range (0, 1] for node {}.".format(
                    j
                )
        for e in self.graph.edges():
            if "L" in self.graph.edges[e]:
                assert (
                    self.graph.edges[e]["L"] >= 0
                ), "The lead time joining nodes {} cannot be negative.".format(e)
            if "p" in self.graph.edges[e]:
                assert (
                    self.graph.edges[e]["p"] >= 0
                ), "The sales price joining nodes {} cannot be negative.".format(e)
            if "b" in self.graph.edges[e]:
                assert (
                    self.graph.edges[e]["b"] >= 0
                ), "The unfulfilled demand costs joining nodes {} cannot be negative.".format(
                    e
                )
            if "g" in self.graph.edges[e]:
                assert (
                    self.graph.edges[e]["g"] >= 0
                ), "The pipeline inventory holding costs joining nodes {} cannot be negative.".format(
                    e
                )
            if "sample_path" in self.graph.edges[e]:
                assert isinstance(
                    self.graph.edges[e]["sample_path"], bool
                ), "When specifying if a user specified demand joining (retailer, market): {} is sampled from a distribution, sample_path must be a Boolean.".format(
                    e
                )
            if "demand_dist" in self.graph.edges[e]:
                dist = self.graph.edges[e]["demand_dist"]  # extract distribution
                assert dist.cdf(
                    0, **self.graph.edges[e]["dist_param"]
                ), "Wrong parameters passed to the demand distribution joining (retailer, market): {}.".format(
                    e
                )
        assert (
            self.backlog == False or self.backlog == True
        ), "The backlog parameter must be a boolean."
        assert (
            self.graph.number_of_nodes() >= 2
        ), "The minimum number of nodes is 2. Please try again"
        assert self.alpha > 0 and self.alpha <= 1, "alpha must be in the range (0, 1]."

        # set random generation seed (unless using user demands)
        self.seed(self.seed_int)

        # action space (reorder quantities for each node for each supplier; list)
        # An action is defined for every node
        num_reorder_links = len(self.reorder_links)
        self.lt_max = np.max(
            [
                self.graph.edges[e]["L"]
                for e in self.graph.edges()
                if "L" in self.graph.edges[e]
            ]
        )
        self.init_inv_max = np.max(
            [
                self.graph.nodes[j]["I0"]
                for j in self.graph.nodes()
                if "I0" in self.graph.nodes[j]
            ]
        )
        self.capacity_max = np.max(
            [
                self.graph.nodes[j]["C"]
                for j in self.graph.nodes()
                if "C" in self.graph.nodes[j]
            ]
        )
        self.pipeline_length = sum(
            [
                self.graph.edges[e]["L"]
                for e in self.graph.edges()
                if "L" in self.graph.edges[e]
            ]
        )
        self.lead_times = {
            e: self.graph.edges[e]["L"]
            for e in self.graph.edges()
            if "L" in self.graph.edges[e]
        }
        self.obs_dim = (
            self.pipeline_length + len(self.main_nodes) + len(self.retail_links)
        )

        # C of factory node
        C = self.graph.nodes[self.factory[0]]["C"]
        Cap = self.graph.nodes[self.factory[0]]["Cap"]
        self.action_space = gym.spaces.Box(
            low=np.zeros(num_reorder_links + len(self.factory)),
            high=np.asarray([Cap] * len(self.distrib) + [C] * len(self.factory)),
            dtype=np.float64,
        )
        print(self.action_space)
        # observation space (total inventory at each node, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=np.ones(self.obs_dim) * np.iinfo(np.int32).min,
            high=np.ones(self.obs_dim) * np.iinfo(np.int32).max,
            dtype=np.float64,
        )
       
        self.edge_list = [
            (i - 1, j - 1)
            for (i, j) in self.reorder_links
            if i in self.nodes and j in self.nodes
        ]
        self.parser = GNNParser(self, T=self.lt_max, edge_list=self.edge_list)
        # intialize
        self.reset()

    def seed(self, seed=None):
        """
        Set random number generation seed
        """
        # seed random state
        if seed != None:
            np.random.seed(seed=int(seed))

    def _RESET(self, demand_params=None):
        """
        Create and initialize all variables and containers.
        Nomenclature:
            I = On hand inventory at the start of each period at each stage (except last one).
            T = Pipeline inventory at the start of each period at each stage (except last one).
            R = Replenishment order placed at each period at each stage (except last one).
            D = Customer demand at each period (at the retailer)
            S = Sales performed at each period at each stage.
            B = Backlog at each period at each stage.
            LS = Lost sales at each period at each stage.
            P = Total profit at each stage.
        """
        T = self.num_periods
        J = len(self.main_nodes)
        RM = len(self.retail_links)  # number of retailer-market pairs
        PS = len(
            self.reorder_links
        )  # number of purchaser-supplier pairs in the network
        SL = len(
            self.network_links
        )  # number of edges in the network (excluding links form raw material nodes)

        # simulation result lists#
        self.X = pd.DataFrame(
            data=np.zeros([T + 12, J]), columns=self.main_nodes
        )  # inventory at the beginning of each period
        self.Y = pd.DataFrame(
            data=np.zeros([T + 12, PS]),
            columns=pd.MultiIndex.from_tuples(
                self.reorder_links, names=["Source", "Receiver"]
            ),
        )  # pipeline inventory at the beginning of each period
        self.R = pd.DataFrame(
            data=np.zeros([T, PS]),
            columns=pd.MultiIndex.from_tuples(
                self.reorder_links, names=["Supplier", "Requester"]
            ),
        )  # replenishment orders
        self.S = pd.DataFrame(
            data=np.zeros([T, SL]),
            columns=pd.MultiIndex.from_tuples(
                self.network_links, names=["Seller", "Purchaser"]
            ),
        )  # units sold
        self.D = pd.DataFrame(
            data=np.zeros([T, RM]),
            columns=pd.MultiIndex.from_tuples(
                self.retail_links, names=["Retailer", "Market"]
            ),
        )  # demand at retailers
        self.U = pd.DataFrame(
            data=np.zeros([T, RM]),
            columns=pd.MultiIndex.from_tuples(
                self.retail_links, names=["Retailer", "Market"]
            ),
        )  # unfulfilled demand for each market - retailer pair
        self.P = pd.DataFrame(
            data=np.zeros([T, J]), columns=self.main_nodes
        )  # profit at each node
        self.Prod = pd.DataFrame(
            data=np.zeros([T, len(self.factory)]), columns=self.factory
        )

        # initializetion
        self.period = 0  # initialize time
        for j in self.main_nodes:
            self.X.loc[0, j] = self.graph.nodes[j]["I0"]  # initial inventory
        self.Y.loc[0, :] = np.zeros(PS)  # initial pipeline inventory
        self.action_log = np.zeros([T, PS])

        if demand_params is not None:
            dmax = demand_params["dmax"]
            r = demand_params["r"]
            f = demand_params["f"]
            self.user_D[(1, 0)], self.demand_rate[(1, 0)] = get_demand_curve(
                dmax=dmax, T=self.num_periods, r=r, f=f
            )
        else:
            if self.version == 1:
                self.user_D[(1, 0)], self.demand_rate[(1, 0)] = get_demand_curve(
                    dmax=25, T=self.num_periods
                )
            if self.version == 2:
              

                dmax = [5, 15, 20]
                r = [1, 3, 6]
                f = [2, 4, 6]

                self.user_D[(1, 0)], self.demand_rate[(1, 0)] = get_demand_curve(
                    dmax=dmax[0],
                    T=self.num_periods,
                    r=r[0],
                    f=f[0],
                    dvar=2,
                )

                self.user_D[(2, 0)], self.demand_rate[(2, 0)] = get_demand_curve(
                    dmax=dmax[1],
                    T=self.num_periods,
                    r=r[1],
                    f=f[1],
                    dvar=2,
                )

                self.user_D[(3, 0)], self.demand_rate[(3, 0)] = get_demand_curve(
                    dmax=dmax[2],
                    T=self.num_periods,
                    r=r[2],
                    f=f[2],
                    dvar=2,
                )

          
            if self.version == 3:
                # 1F10S
                dmax = [5] * 4 + [10] * 3 + [18] * 3
                
                r = [1, 1, 1, 3, 3, 3, 6, 6, 6, 2]
                f = [2, 4, 6, 2, 4, 6, 2, 4, 6, 3]

                self.user_D[(1, 0)], self.demand_rate[(1, 0)] = get_demand_curve(
                    dmax=dmax[0], T=self.num_periods, r=r[0], f=f[0]
                )

                self.user_D[(2, 0)], self.demand_rate[(2, 0)] = get_demand_curve(
                    dmax=dmax[1], T=self.num_periods, r=r[1], f=f[1]
                )

                self.user_D[(3, 0)], self.demand_rate[(3, 0)] = get_demand_curve(
                    dmax=dmax[2], T=self.num_periods, r=r[2], f=f[2]
                )

                self.user_D[(4, 0)], self.demand_rate[(4, 0)] = get_demand_curve(
                    dmax=dmax[3], T=self.num_periods, r=r[3], f=f[3]
                )
                self.user_D[(5, 0)], self.demand_rate[(5, 0)] = get_demand_curve(
                    dmax=dmax[4], T=self.num_periods, r=r[4], f=f[4]
                )
                self.user_D[(6, 0)], self.demand_rate[(6, 0)] = get_demand_curve(
                    dmax=dmax[5], T=self.num_periods, r=r[5], f=f[5]
                )
                self.user_D[(7, 0)], self.demand_rate[(7, 0)] = get_demand_curve(
                    dmax=dmax[6], T=self.num_periods, r=r[6], f=f[6]
                )
                self.user_D[(8, 0)], self.demand_rate[(8, 0)] = get_demand_curve(
                    dmax=dmax[7], T=self.num_periods, r=r[7], f=f[7]
                )
                self.user_D[(9, 0)], self.demand_rate[(9, 0)] = get_demand_curve(
                    dmax=dmax[8], T=self.num_periods, r=r[8], f=f[8]
                )
                self.user_D[(10, 0)], self.demand_rate[(10, 0)] = get_demand_curve(
                    dmax=dmax[9], T=self.num_periods, r=r[9], f=f[9]
                )
        
        for j in self.retail:
            for k in self.market:
                # read user specified demand. if all zeros, use demand_dist instead.
                for t in range(T):
                    # Demand = self.graph.edges[(j, k)]["user_D"]
                    Demand = self.user_D[(j, k)]
                    if np.sum(Demand) > 0:
                        self.D.loc[t, (j, k)] = Demand[t]
                    else:
                        Demand = self.graph.edges[(j, k)]["demand_dist"]
                        self.D.loc[t, (j, k)] = Demand.rvs(
                            **self.graph.edges[(j, k)]["dist_param"]
                        )

        # set state
        self.period = 1
        if self.use_mlp:
            self._update_state_mlp()
        else:
            self._update_state()

        return self.state

    def _update_state(self):
        pipeline = np.zeros((len(self.nodes), self.lt_max))
        for i, j in self.Y.keys():
            pipeline[j - 1] += self.Y[i, j][
                self.period : self.period + self.lt_max
            ].values
        for n in self.factory:
            pipeline[n - 1] = self.Prod.loc[self.period - 1, n]

        self.state = self.parser.parse_obs(pipeline)

    def _STEP(self, prod_action, distr_action):
        """
        Take a step in time in the multiperiod inventory management problem.
        prod_action: number of units to produce at each factory node.
        distr_action: number of units to distribute from each supplier to each purchaser.
        """
        t = self.period
      
        for j in self.factory:
            self.X.loc[t - 1, j] += self.Prod.loc[t - 1, j]
        X_temp = {}
        for i in self.main_nodes:
            X_temp[i] = self.X.loc[t - 1, i]
        for req in distr_action:
            supplier = req[0]
            purchaser = req[1]
            X_temp[supplier] -= distr_action[req]

        # check how many negative stocks
        over_order = 0
        for i in self.main_nodes:
            if X_temp[i] < 0:
                over_order += np.abs(X_temp[i])

        # Place Orders
        X_factory = {}
        for supplier in self.main_nodes:
            X_factory[supplier] = self.X.loc[t - 1, supplier]

        produ_c = {}
        for supplier in self.factory:
            produ_c[supplier] = self.graph.nodes[supplier]["C"]

        for key in prod_action.keys():
            request = round(max(prod_action[key], 0))  # force to integer value
            production = key
            request = min(produ_c[production], request)
            produ_c[production] -= request
            self.Prod.loc[t, production] = request

        l = distr_action.keys()
        l = random.sample(l, len(l))
     
        for key in l:
            request = round(max(distr_action[key], 0))  # force to integer value
            supplier = key[0]
            purchaser = key[1]

            if supplier in self.distrib:
                X_supplier = X_factory[supplier]
              
                # request limited by available inventory at beginning of period and remaining capacity of purchaser
                request = min(
                    request,
                    X_supplier,
                   
                )
                
                self.R.loc[t, (supplier, purchaser)] = request
                self.S.loc[t, (supplier, purchaser)] = request

                X_factory[supplier] -= self.S.loc[t, (supplier, purchaser)]
               
            elif supplier in self.factory:
                
                v = self.graph.nodes[supplier]["v"]  # supplier yield
           
                request = min(
                    request,
                    v * X_factory[supplier],
                    
                )
            
                self.R.loc[t, (supplier, purchaser)] = request
                self.S.loc[t, (supplier, purchaser)] = request
                produ_c[supplier] -= self.S.loc[t, (supplier, purchaser)]

                X_factory[supplier] -= self.S.loc[t, (supplier, purchaser)] / v
            
    
        # Receive deliveries and update inventories
        for j in self.main_nodes:
            # update pipeline inventories

            if j not in self.factory:
                incoming = []
                for k in self.graph.predecessors(j):
                    L = self.graph.edges[(k, j)]["L"]  # extract lead time
                    if t - L >= 1:  # check if delivery has arrived
                        delivery = self.R.loc[t - L, (k, j)]
                    else:
                        delivery = 0
                    incoming += [delivery]  # update incoming material
                  
                    self.Y.loc[t + L, (k, j)] += self.R.loc[t, (k, j)]
                # update on-hand inventory
                if "v" in self.graph.nodes[j]:  # extract production yield
                    v = self.graph.nodes[j]["v"]
                else:
                    v = 1
                outgoing = (
                    1
                    / v
                    * np.sum([self.S.loc[t, (j, k)] for k in self.graph.successors(j)])
                )  # consumed inventory (for requests placed)

                self.X.loc[t, j] = self.X.loc[t - 1, j] + np.sum(incoming) - outgoing

        for j in self.factory:
            outgoing = np.sum([self.S.loc[t, (j, k)] for k in self.graph.successors(j)])

            self.X.loc[t, j] = self.X.loc[t - 1, j] - outgoing

        # demand is realized
        for j in self.retail:
            for k in self.market:
             
                if self.backlog and t >= 1:
                    D = self.D.loc[t, (j, k)] + self.U.loc[t - 1, (j, k)]
                else:
                    D = self.D.loc[t, (j, k)]
                # satisfy demand up to available level
                X_retail = self.X.loc[
                    t, j
                ]  # get inventory at retail before demand was realized
                self.S.loc[t, (j, k)] = min(D, X_retail)  # perform sale
                self.X.loc[t, j] -= self.S.loc[t, (j, k)]  # update inventory
                self.U.loc[t, (j, k)] = (
                    D - self.S.loc[t, (j, k)]
                )  # update unfulfilled orders

        for j in self.nodes:
            assert (
                self.X.loc[t, j] >= 0
            ), f"Negative inventory at node {j} with inventory {self.X.loc[t, j]}"

        # calculate profit
        sales_revenue = 0
        purchasing_costs = 0
        transportation_costs = 0
        operating_costs = 0
        holding_costs = 0
        unfulfilled_penalty = 0
        capacity_penalty = 0
        violated_cap = 0
        for j in self.main_nodes:
            a = self.alpha
            SR = np.sum(
                [
                    self.graph.edges[(j, k)]["p"] * self.S.loc[t, (j, k)]
                    for k in self.graph.successors(j)
                ]
            )  # sales revenue
            PC = np.sum(
                [
                    self.graph.edges[(k, j)]["p"] * self.R.loc[t, (k, j)]
                    for k in self.graph.predecessors(j)
                ]
            )  # purchasing costs
            TC = np.sum(
                [
                    self.graph.edges[(k, j)]["g"]
                    * self.R.loc[t, (k, j)]
                    * self.lead_times[(k, j)]
                    for k in self.graph.predecessors(j)
                ]
            )
            # transportation cost

            HC = self.graph.nodes[j]["h"] * self.X.loc[t, j]
       
            if j in self.factory:
                OC = (
                    self.graph.nodes[j]["o"]
                    / self.graph.nodes[j]["v"]
                    * self.Prod.loc[t, j]
                )  # operating costs
            else:
                OC = 0
            if j in self.retail:
                UP = np.sum(
                    [
                        self.graph.edges[(j, k)]["b"] * self.U.loc[t, (j, k)]
                        for k in self.graph.successors(j)
                    ]
                )  # unfulfilled penalty
            else:
                UP = 0

            # capacity penalty
            if self.X.loc[t, j] > self.graph.nodes[j]["Cap"]:
                CP = (self.X.loc[t, j] - self.graph.nodes[j]["Cap"]) * 1.5 * 15
                violated_cap += self.X.loc[t, j] - self.graph.nodes[j]["Cap"]
            else:
                CP = 0

            self.P.loc[t, j] = a**t * (SR - PC - OC - HC - UP - TC - CP)
            sales_revenue += SR
            purchasing_costs += PC
            transportation_costs += TC
            operating_costs += OC
            holding_costs += HC
            unfulfilled_penalty += UP
            capacity_penalty += CP

        info = {
            "sales_revenue": sales_revenue,
            "purchasing_costs": purchasing_costs,
            "transportation_costs": transportation_costs,
            "operating_costs": operating_costs,
            "holding_costs": holding_costs,
            "unfulfilled_penalty": unfulfilled_penalty,
            "unfulfilled_demand": np.sum(self.U.loc[t, :]),
            "demand": D,
            "sold units": self.S.loc[t, (1, 0)],
            "capacity_penalty": capacity_penalty,
            "violated_capacity": violated_cap,
            "over_order": over_order,
        }

        # update period
        self.period += 1
        # set reward (profit from current timestep)
        reward = self.P.loc[t, :].sum()
        reward -= over_order * 6
  
        # determine if simulation should terminate
        if self.period >= self.num_periods:
            done = True
        else:
            done = False
            # update stae
            if self.use_mlp:
                self._update_state_mlp()
            else:
                self._update_state()

        return self.state, reward, done, info

    def sample_action(self):
        """
        Generate an action by sampling from the action_space
        """
        return self.action_space.sample()

    def step(self, prod_action, distr_action):
        return self._STEP(prod_action, distr_action)

    def reset(self, seed=None, options=None):
        return self._RESET()

class NetInvMgmtBacklogEnv(NetInvMgmtMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class NetInvMgmtLostSalesEnv(NetInvMgmtMasterEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backlog = False
