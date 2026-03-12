import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from optimizers import optimizer as opt
import data_generation as dg
import time


def obj_fun(Y, X, Beta_k, Omega_k, lmbda_b, lmbda_o):
    n = Y.shape[0]
    R = Y - X @ Beta_k

    sign, logdet = np.linalg.slogdet(Omega_k)

    ll = (n / 2) * (np.trace((R.T @ R @ Omega_k) / n) - logdet)
    pen_b = lmbda_b * np.sum(np.abs(Beta_k))
    pen_o = lmbda_o * (np.sum(np.abs(Omega_k)) - np.sum(np.abs(np.diag(Omega_k))))
    return ll + pen_b + pen_o


def confusion_matrix(real, estimate, tolerance, file, B=True):
    tp = np.sum((real == 0) & (np.abs(estimate) < tolerance))
    tn = np.sum((real != 0) & (np.abs(estimate) > tolerance))
    fp = np.sum((real != 0) & (np.abs(estimate) < tolerance))
    fn = np.sum((real == 0) & (np.abs(estimate) > tolerance))
    cm = pd.DataFrame(
        index=["Actual Positive", "Actual Negative"],
        data={"Pred_Pos": [tp, fp], "Pred_Neg": [fn, tn]},
    )
    with open(file, "a") as file:
        file.write("\n")
        file.write(
            f"Confusion Mat for {"Beta" if B else "Omega"} as Threshold: {tolerance} \n"
        )
        file.write(cm.to_string())
        file.write("\n")


def train(
    data,
    lmbda_b,
    lmbda_o,
    problem=None,
    rho=None,
    neighberhood=True,
    epsilon=1e-3,
    max_iter=1000,
    hp=False,
):

    start_time = time.perf_counter()
    Y, X, B, Omega = data["Y"], data["X"], data["B"], data["Omega"]
    B_k = np.zeros(B.shape)
    Omega_k = np.eye(Omega.shape[0])
    Gamma_k = np.eye(Omega.shape[0])
    U_k = np.zeros_like(Omega)

    iter = 0
    loss = [np.inf, 0.0]

    while ((np.abs(loss[-2] - loss[-1]) > epsilon)) & (iter < max_iter):

        B_k = opt.beta_cd(Y, X, B_k, Omega_k, lmbda_b)

        if neighberhood:
            Omega_k = opt.neighberhood_selection_cd(Y, X, B_k, lmbda_o)
        else:
            Omega_k, Gamma_k, U_k = opt.ADMM(
                Y, X, Omega_k, Gamma_k, U_k, B_k, rho, 2 * lmbda_o / Y.shape[0], 100
            )
        loss.append(obj_fun(Y, X, B_k, Omega_k, lmbda_b, lmbda_o))
        iter += 1
        if not hp:
            print(f"Iter : {iter}, Loss : {loss[-1]}")

    b_rel_err = (np.linalg.matrix_norm(B_k - B, ord="fro") ** 2) / (
        np.linalg.matrix_norm(B, ord="fro") ** 2
    )

    o_rel_err = (np.linalg.matrix_norm(Omega_k - Omega, ord="fro") ** 2) / (
        np.linalg.matrix_norm(Omega, ord="fro") ** 2
    )

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    if not hp:
        with open(f"{problem.strip()}.txt", "w") as file:
            file.write(f"Results for {problem} \n")
            file.write(f"Total Iterations: {iter} \n")
            file.write(f"Total Time: {elapsed_time} seconds \n")
            file.write(f"Final Loss : {loss[-1]} \n")
            file.write(f"Beta Relative Error : {b_rel_err} \n")
            file.write(f"Omega Relative Error : {o_rel_err} \n")

    if not hp:
        matrix_diagnostics(B, B_k, Omega, Omega_k, loss, 1e-4, problem)
    return np.mean([b_rel_err, o_rel_err])


def matrix_diagnostics(B, B_k, Omega, Omega_k, loss, threshold, problem):
    # Beta Diagnostics
    print(f"Beta Confusion Matrix at Threshold : {threshold}:")
    confusion_matrix(B, B_k, 1e-4, f"{problem.strip()}.txt", B=True)
    dg.plot_precision_matrix(B - B_k, problem, B=True)
    PR_curve(B, B_k, problem, B=True)
    ROC_curve(B, B_k, problem, B=True)

    # Omega Diagnostics
    print(f"Omega Confusion Matrix at Threshold : {threshold}:")
    confusion_matrix(Omega, Omega_k, 1e-4, f"{problem.strip()}.txt", B=False)
    dg.plot_precision_matrix(Omega - Omega_k, problem, B=False)
    PR_curve(Omega, Omega_k, problem, B=False)
    ROC_curve(Omega, Omega_k, problem, B=False)

    # Objective Diagnostics
    plt.figure()
    ax = sns.scatterplot(x=range(len(loss[2:])), y=loss[2:])
    ax.set_title(problem)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    fig = ax.get_figure()
    title = f"Objective {problem.strip()}"
    fig.savefig(title, dpi=300, bbox_inches="tight")


def PR_curve1(parameter, estimate, problem, B):
    precisions, recalls = [], []
    unique_vals = np.sort(np.unique(np.abs(estimate)))
    for tolerance in unique_vals:
        tp = np.sum((parameter == 0) & (np.abs(estimate) <= tolerance))
        tn = np.sum((parameter != 0) & (np.abs(estimate) > tolerance))
        fp = np.sum((parameter != 0) & (np.abs(estimate) <= tolerance))
        fn = np.sum((parameter == 0) & (np.abs(estimate) > tolerance))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls.append(recall)
        precisions.append(precision)

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    if not B:
        print(recalls)
        print(precisions)

    fig = plt.figure()
    sns.lineplot(x=recalls, y=precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    title = f"PR Curve for {'Beta' if B else 'Omega'} for {problem}"
    plt.title(title)
    title = f"PR{title.strip()}"
    fig.savefig(title, dpi=300, bbox_inches="tight")


def PR_curve(parameter, estimate, problem, B):

    # Note chatGPt helped with this code as my PR curve kept having a fork

    curve_name = "Beta" if B else "Omega"

    if B:
        param_vec = parameter.ravel()
        est_vec = estimate.ravel()
    else:
        # upper triangle only, exclude diagonal
        mask = np.triu(np.ones_like(parameter, dtype=bool), k=1)
        param_vec = parameter[mask]
        est_vec = estimate[mask]

    y_true = (param_vec == 0).astype(int)

    score = np.abs(est_vec)
    order = np.argsort(score)
    y_true = y_true[order]
    score = score[order]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    total_pos = y_true.sum()

    recall = tp / total_pos if total_pos > 0 else np.zeros_like(tp, dtype=float)
    precision = tp / (tp + fp)

    # collapse tied scores so each threshold contributes one point
    last_in_tie_block = np.r_[np.where(np.diff(score) != 0)[0], len(score) - 1]
    recalls = recall[last_in_tie_block]
    precisions = precision[last_in_tie_block]

    pairs = [(r, p) for r, p in zip(recalls, precisions) if r > 0.01]

    if pairs:
        recalls_plot, precisions_plot = zip(*pairs)
    else:
        recalls_plot, precisions_plot = recalls, precisions

    baseline = y_true.mean()

    fig = plt.figure()
    plt.plot(recalls_plot, precisions_plot, marker="o")
    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f"Baseline = {baseline:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve for Zero Entries in {curve_name} for {problem}")
    plt.legend()

    safe_problem = problem.strip().replace(" ", "_").replace(",", "")
    fig.savefig(
        f"PR_zero_{curve_name}_{safe_problem}.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def ROC_curve(parameter, estimate, problem, B):

    curve_name = "Beta" if B else "Omega"

    if B:
        param_vec = parameter.ravel()
        est_vec = estimate.ravel()
    else:
        # upper triangle only, exclude diagonal
        mask = np.triu(np.ones_like(parameter, dtype=bool), k=1)
        param_vec = parameter[mask]
        est_vec = estimate[mask]

    # Positive class = true zeros
    y_true = (param_vec == 0).astype(int)

    # Smaller |estimate| = more likely zero
    score = np.abs(est_vec)
    order = np.argsort(score)
    y_true = y_true[order]
    score = score[order]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    tpr = tp / total_pos if total_pos > 0 else np.zeros_like(tp, dtype=float)
    fpr = fp / total_neg if total_neg > 0 else np.zeros_like(fp, dtype=float)

    # collapse tied scores so each threshold contributes one point
    last_in_tie_block = np.r_[np.where(np.diff(score) != 0)[0], len(score) - 1]
    tpr = tpr[last_in_tie_block]
    fpr = fpr[last_in_tie_block]

    # include endpoints
    fpr_plot = np.r_[0, fpr, 1]
    tpr_plot = np.r_[0, tpr, 1]

    fig = plt.figure()
    plt.plot(fpr_plot, tpr_plot, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Zero Entries in {curve_name} for {problem}")

    safe_problem = problem.strip().replace(" ", "_").replace(",", "")
    fig.savefig(
        f"ROC_zero_{curve_name}_{safe_problem}.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
