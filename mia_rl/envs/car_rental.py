import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass(frozen=True)
class CarRentalParams:
    max_cars_1: int = 20
    max_cars_2: int = 20
    max_moveable: int = 5

    revenue_per_rental: float = 10.0
    cost_per_moved: float = 2.0

    lambdas: Tuple[float, float, float, float] = (
        3.0, 4.0, 3.0, 2.0
    )

    max_requests_1: int = 8
    max_requests_2: int = 10
    max_returns_1: int = 8
    max_returns_2: int = 8


def poisson_pmf_truncated(lam: float, max_k: int):
    probs = np.zeros(max_k + 1)

    probs[0] = np.exp(-lam)

    for k in range(1, max_k):
        probs[k] = probs[k - 1] * lam / k

    probs[max_k] = 1.0 - probs[:max_k].sum()

    probs /= probs.sum()

    return probs


class CarRentalMDP:

    def __init__(self, params: CarRentalParams):

        self.params = params

        self.req1 = poisson_pmf_truncated(
            params.lambdas[0],
            params.max_requests_1,
        )

        self.req2 = poisson_pmf_truncated(
            params.lambdas[1],
            params.max_requests_2,
        )

        self.ret1 = poisson_pmf_truncated(
            params.lambdas[2],
            params.max_returns_1,
        )

        self.ret2 = poisson_pmf_truncated(
            params.lambdas[3],
            params.max_returns_2,
        )

        self._loc_cache = {}

    def states(self):

        return [
            (i, j)
            for i in range(self.params.max_cars_1 + 1)
            for j in range(self.params.max_cars_2 + 1)
        ]

    def possible_actions(self, s):

        n1, n2 = s

        a_min = -min(
            self.params.max_moveable,
            n2,
            self.params.max_cars_1 - n1,
        )

        a_max = min(
            self.params.max_moveable,
            n1,
            self.params.max_cars_2 - n2,
        )

        return list(range(a_min, a_max + 1))

    def after_move(self, s, a):

        n1, n2 = s

        return (n1 - a, n2 + a)

    def _loc_outcomes(self, loc_id, cars_after_move):

        key = (loc_id, cars_after_move)

        if key in self._loc_cache:
            return self._loc_cache[key]

        if loc_id == 1:
            req = self.req1
            ret = self.ret1
            cap = self.params.max_cars_1
        else:
            req = self.req2
            ret = self.ret2
            cap = self.params.max_cars_2

        p_next = np.zeros(cap + 1)
        exp_rented = 0.0

        for k_req, p_req in enumerate(req):

            rented = min(cars_after_move, k_req)

            exp_rented += p_req * rented

            cars_left = cars_after_move - rented

            for k_ret, p_ret in enumerate(ret):

                next_cars = min(cap, cars_left + k_ret)

                p_next[next_cars] += p_req * p_ret

        p_next /= p_next.sum()

        self._loc_cache[key] = (p_next, exp_rented)

        return p_next, exp_rented

    def expected_transition(self, s, a):

        if a not in self.possible_actions(s):
            raise ValueError("Illegal action")

        n1m, n2m = self.after_move(s, a)

        p1, e1 = self._loc_outcomes(1, n1m)
        p2, e2 = self._loc_outcomes(2, n2m)

        exp_revenue = (
            e1 + e2
        ) * self.params.revenue_per_rental

        return p1, p2, exp_revenue