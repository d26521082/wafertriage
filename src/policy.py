"""Station 2.3: evaluate the three-way triage policy on validation data.

Thresholds (derived analytically in station 2.2, C_rev = 1 as numeraire):
    p1 = C_rev / (C_miss - C_inv)      below p1: auto-clear
    p2 = (C_inv - C_rev) / C_inv       above p2: auto-flag; between: human review

For each C_miss on a sweep grid, report band sizes, realized expected cost per
wafer, residual risk among auto-cleared wafers, and cost of baseline policies:
    all-review : every wafer goes to a human
    all-flag   : every wafer triggers an investigation
    argmax     : no triage — clear if predicted none, else flag
"""

import numpy as np

from calibration import get_probs

C_REV = 1.0
C_INV = 10.0
C_MISS_GRID = [10, 30, 100, 300, 1000]


def main() -> None:
    probs, targets = get_probs()
    p = 1.0 - probs[:, 0]          # P(defect) = 1 - P(none)
    defect = targets != 0          # ground truth on val
    pred_defect = probs.argmax(axis=1) != 0
    n = len(p)
    print(f"val wafers: {n:,}, true defect share: {defect.mean():.3%}\n")

    print(f"{'C_miss':>7}{'p1':>9}{'p2':>6} | {'clear':>7}{'review':>7}{'flag':>6} | "
          f"{'policy':>8}{'all-rev':>8}{'all-flag':>9}{'argmax':>8} | {'miss@clear':>11}")
    for c_miss in C_MISS_GRID:
        if c_miss <= C_INV:
            # missing costs no more than investigating: triage degenerates to
            # "clear everything" — nothing is ever worth looking at
            p1, p2 = 1.0, 1.0
        else:
            p1 = min(1.0, C_REV / (c_miss - C_INV))
            p2 = (C_INV - C_REV) / C_INV

        clear = p < p1
        flag = p >= p2
        review = ~clear & ~flag

        cost = np.where(clear, np.where(defect, c_miss, 0.0),
               np.where(flag, C_INV,
                        C_REV + np.where(defect, C_INV, 0.0)))
        cost_all_review = (C_REV + C_INV * defect).mean()
        cost_all_flag = C_INV
        cost_argmax = np.where(pred_defect, C_INV,
                               np.where(defect, c_miss, 0.0)).mean()

        missed = int((clear & defect).sum())
        print(f"{c_miss:>7}{p1:>9.4f}{p2:>6.2f} | "
              f"{clear.mean():>6.1%}{review.mean():>7.1%}{flag.mean():>6.1%} | "
              f"{cost.mean():>8.3f}{cost_all_review:>8.3f}{cost_all_flag:>9.3f}"
              f"{cost_argmax:>8.3f} | {missed:>4d} ({missed / max(clear.sum(),1):.2%})")


if __name__ == "__main__":
    main()
