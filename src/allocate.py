"""Station 2.4: inspection-capacity allocation with lot-opening costs.

Setting: the C_miss=100 triage sends ~24% of wafers to human review, but review
time is budgeted. Reviewing wafer i takes 1 unit; opening its lot (pulling lot
history / tool logs) costs F units once per lot. Unreviewed wafers fall back to
the cheaper of auto-clear / auto-flag.

    max  sum b_i x_i
    s.t. F * sum y_j + sum x_i <= K
         x_i <= y_lot(i)
         x, y binary

Solved exactly with HiGHS (scipy.optimize.milp) and compared against the naive
greedy (rank by b_i, open lots as you go) to price the value of optimisation.
"""

import numpy as np
import h5py
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import lil_matrix

from calibration import get_probs
from dataset import H5, SPLITS

C_REV, C_INV, C_MISS = 1.0, 10.0, 100.0
F = 5.0                      # lot-opening cost, in units of one wafer review
K_GRID = [1000, 3000, 6000, 10000]


def review_band_and_benefit():
    probs, _ = get_probs()
    p = 1.0 - probs[:, 0]
    p1 = C_REV / (C_MISS - C_INV)
    p2 = (C_INV - C_REV) / C_INV
    band = (p >= p1) & (p < p2)

    b = np.minimum(p * C_MISS, C_INV) - (C_REV + p * C_INV)   # benefit of review
    val_idx = np.load(SPLITS)["val"]
    with h5py.File(H5, "r") as f:
        lots_all = f["lot"].asstr()[:]
    lots = lots_all[val_idx]
    return p[band], b[band], lots[band]


def solve_milp(b, lot_id, K):
    n = len(b)
    lots = np.unique(lot_id)
    m = len(lots)
    lot_pos = {l: j for j, l in enumerate(lots)}

    c = np.concatenate([-b, np.zeros(m)])          # milp minimises; negate benefit

    budget = np.concatenate([np.ones(n), np.full(m, F)])
    cons = [LinearConstraint(budget, -np.inf, K)]

    link = lil_matrix((n, n + m))
    for i in range(n):
        link[i, i] = 1.0
        link[i, n + lot_pos[lot_id[i]]] = -1.0     # x_i - y_lot(i) <= 0
    cons.append(LinearConstraint(link.tocsc(), -np.inf, 0.0))

    res = milp(c=c, constraints=cons, integrality=np.ones(n + m),
               bounds=(0, 1), options={"time_limit": 120})
    x = res.x[:n] > 0.5
    y = res.x[n:] > 0.5
    return b[x].sum(), int(x.sum()), int(y.sum()), res.status


def solve_greedy(b, lot_id, K):
    order = np.argsort(-b)
    spent, benefit, reviewed = 0.0, 0.0, 0
    open_lots = set()
    for i in order:
        cost = 1.0 + (F if lot_id[i] not in open_lots else 0.0)
        if spent + cost > K:
            continue
        spent += cost
        benefit += b[i]
        reviewed += 1
        open_lots.add(lot_id[i])
    return benefit, reviewed, len(open_lots)


def main() -> None:
    p, b, lot_id = review_band_and_benefit()
    n_lots = len(np.unique(lot_id))
    full_cost = len(b) + F * n_lots
    print(f"review band: {len(b):,} wafers across {n_lots:,} lots")
    print(f"total benefit if everything reviewed: {b.sum():,.0f} "
          f"(would cost {full_cost:,.0f} units)\n")

    print(f"{'K':>7} | {'MIP benefit':>12}{'wafers':>8}{'lots':>6} | "
          f"{'greedy':>9}{'wafers':>8}{'lots':>6} | {'MIP edge':>9}")
    for K in K_GRID:
        mb, mw, ml, status = solve_milp(b, lot_id, K)
        gb, gw, gl = solve_greedy(b, lot_id, K)
        edge = (mb - gb) / gb if gb > 0 else float("inf")
        print(f"{K:>7,} | {mb:>12,.1f}{mw:>8,}{ml:>6,} | "
              f"{gb:>9,.1f}{gw:>8,}{gl:>6,} | {edge:>8.1%}")


if __name__ == "__main__":
    main()
