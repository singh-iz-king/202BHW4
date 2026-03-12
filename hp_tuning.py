import numpy as np
from optimizers import train
from data_generation import generate_data


def hyper_parameter_grid_search(
    data,
    problem,
    admm=True,
    lambda_b_vals=[1, 5, 20],
    lambda_o_vals=[1, 5, 20],
    rho=[0.25, 0.5, 0.75, 1],
):

    best_loss = np.inf
    best = [0, 0, 0]

    with open(f"{problem.strip()}hp.txt", "a") as f:

        f.write(f"Hyperparameter Grid Search for {problem} \n")

        for lb in lambda_b_vals:
            for lo in lambda_o_vals:
                if admm:
                    for r in rho:
                        loss = train.train(
                            data=data,
                            lmbda_b=lb,
                            lmbda_o=lo,
                            rho=r,
                            neighberhood=False,
                            epsilon=1e-3,
                            max_iter=10,
                            hp=True,
                        )
                        if loss < best_loss:
                            best = [lb, lo, r]
                            best_loss = loss
                        f.write(
                            f"Loss : {loss} for Lambda_B = {lb}, Lambda_O = {lo}, Rho = {r} \n"
                        )
                else:
                    loss = train.train(
                        data=data,
                        lmbda_b=lb,
                        lmbda_o=lo,
                        neighberhood=True,
                        epsilon=1e-3,
                        max_iter=10,
                        hp=True,
                    )
                    if loss < best_loss:
                        best = [lb, lo, None]
                        best_loss = loss
                    f.write(f"Loss : {loss} for Lambda_B = {lb}, Lambda_O = {lo} \n")
            print(
                f"Best So Far Lambda_B = {best[0]}, Lambda_O = {best[1]}, Rho = {best[2]} with Loss {best_loss}"
            )
        f.write("\n")
        f.write(
            f"Best: Lambda_B = {best[0]}, Lambda_O = {best[1]}, Rho = {best[2]} with Loss {best_loss} \n"
        )

    return best
