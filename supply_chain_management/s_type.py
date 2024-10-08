from collections import defaultdict

class s_type_policy:
    def __init__(self, factory_s, warehouses_s):
        self.factory_s = factory_s
        self.warehouses_s = warehouses_s

    def select_action(self, env):

        factory = env.factory[0]
        cap = env.graph.nodes[(factory)]["C"]
        # compute the desired shipping quantities
        ship = dict()
        desiredStoreOrder = defaultdict()
        desiredStoreOrder_sum = 0
        # get desired shipping quantities for all stores

        for i, j in env.reorder_links:
            if j in env.distrib:
                desiredStoreOrder[(i, j)] = self.warehouses_s[env.distrib.index(j)] - (
                    env.X[j][env.period - 1]
                    + env.Y[i, j][env.period : env.period + env.lt_max].sum()
                    - (
                        env.D.loc[env.period, (j, 0)]
                        + env.U.loc[env.period - 1, (j, 0)]
                    )
                )
                desiredStoreOrder[(i, j)] = max(0, desiredStoreOrder[(i, j)])
                desiredStoreOrder_sum += desiredStoreOrder[(i, j)]
        # if all store orders are feasible under the current factory availability: execute it

        if env.X[factory][env.period - 1] >= desiredStoreOrder_sum:
            ship = desiredStoreOrder

        # otherwise, select store orders to maximize the minimum inventoy among all stores

        else:
            ratios = [
                (
                    desiredStoreOrder[i, j] / desiredStoreOrder_sum
                    if j in env.distrib
                    else None
                )
                for i, j in env.reorder_links
            ]
            ship = dict()
            for i, key in enumerate(desiredStoreOrder.keys()):
                ship[key] = int(env.X[factory][env.period - 1] * ratios[i])

        prod = dict()
        # compute available products at factory nodes
        av_factory = dict()

        av_factory[factory] = (
            env.X[factory][env.period - 1]
            + env.Prod[factory][env.period - 1]
            - sum([ship[key] for key in ship if key[0] == factory])
        )

        diff = self.factory_s - av_factory[factory]
        diff = max(0, diff)

        prod[factory] = min(diff, cap)

        return prod, ship
