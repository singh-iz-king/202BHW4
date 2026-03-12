import numpy as np
import sys

from optimizers import hp_tuning, train
from data_generation import generate_data


def main():
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except:
            print("Invalid Intget provided for seed, using default seed 42")
            seed = 42
        print(f"Random Seed: {seed}")
    else:
        print("No Random seed provided, using default seed 42")

    rng = np.random.default_rng(seed)

    data1 = generate_data(200, 50, 50, 0.02, 0.05, rng)
    data2 = generate_data(500, 100, 50, 0.05, 0.1, rng)

    # Problem 1
    problem1 = "N = 200, P = 50, Neighberhood"

    print(problem1)

    params = hp_tuning.hyper_parameter_grid_search(data1, problem1, admm=False)

    train.train(
        data1,
        lmbda_b=params[0],
        lmbda_o=params[1],
        rho=None,
        problem=problem1,
        neighberhood=True,
        epsilon=1e-6,
        max_iter=100,
        hp=False,
    )

    # Problem 2
    problem2 = "N = 500, P = 100, Neighberhood"

    print(problem2)

    params = hp_tuning.hyper_parameter_grid_search(data2, problem2, admm=False)

    train.train(
        data2,
        lmbda_b=params[0],
        lmbda_o=params[1],
        rho=None,
        problem=problem2,
        neighberhood=True,
        epsilon=1e-6,
        max_iter=100,
        hp=False,
    )

    # Problem 3
    problem3 = "N = 200, P = 50, ADMM"

    print(problem3)

    params = hp_tuning.hyper_parameter_grid_search(data1, problem3, admm=True)

    train.train(
        data1,
        lmbda_b=params[0],
        lmbda_o=params[1],
        rho=params[2],
        problem=problem3,
        neighberhood=False,
        epsilon=1e-6,
        max_iter=100,
        hp=False,
    )

    # Problem 4
    problem4 = "N = 500, P = 100, ADMM"

    print(problem4)

    params = hp_tuning.hyper_parameter_grid_search(data2, problem4, admm=True)

    train.train(
        data2,
        lmbda_b=params[0],
        lmbda_o=params[1],
        rho=params[2],
        problem=problem4,
        neighberhood=False,
        epsilon=1e-6,
        max_iter=100,
        hp=False,
    )


if __name__ == "__main__":
    main()
