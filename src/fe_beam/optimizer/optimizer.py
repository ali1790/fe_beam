import sys
from os import path
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from fe_beam.optimizer.objectives import FatigueTestSingle


def denormalize(pop01, bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + pop01 * (hi - lo)

def log_population_csv(filename_all, filename_best, gen, pop, fvals, min_index):
    # gen: int
    # pop: (NP, dim)
    # fvals: (NP,)
    with open(filename_all, "a", encoding="utf-8") as f:
        for i in range(pop.shape[0]):
            f.write(
                f"{gen},{i},{fvals[i]:.12e}," +
                ",".join(f"{v:.12e}" for v in pop[i]) +
                "\n"
            )
    with open(filename_best, "a", encoding="utf-8") as f:
        f.write(f"{gen},{fvals[min_index]:.12e}," +",".join(f"{v:.12e}" for v in pop[min_index]) +
            "\n"
        )

def run_optimzation(objective: FatigueTestSingle, bounds: List[Tuple] , parameters: Dict, log_dir: str):
    solver = DifferentialEvolutionSolver(
        func=objective,
        bounds = bounds,
        #popsize = parameters['popsize'],
        #maxiter = parameters['maxiter'],
        workers = -1,#parameters['workers'],
        updating='deferred'
    )

    gen = 0

    t = datetime.now()
    date_str = f'{t.year}_{t.month}_{t.day}__{t.hour}_{t.minute}'
    log_file_all = path.join(log_dir, f'All_results_{date_str}.csv' )
    log_file_best = path.join(log_dir, f'Best_results_{date_str}.csv' )

    with open(log_file_all, "w", encoding="utf-8") as f:
        f.write("generation,individual,fx," +
                ",".join([f"x{j}" for j in range(len(bounds))]) +
                "\n")
    with open(log_file_best, "w", encoding="utf-8") as f:
        f.write("generation,fx," +
                ",".join([f"x{j}" for j in range(len(bounds))]) +
                "\n")

    try:
        # Iterator: jeder next()-Schritt entspricht einer Generation
        # (StopIteration, wenn maxiter erreicht/konvergiert)
        while not solver.converged():
            print(f'Iteration: {gen}')
            next(solver)  # eine Generation weiter

            pop_normalized = np.asarray(solver.population)
            pop =  denormalize( pop_normalized,  bounds )
            fvals =  solver.population_energies

            log_population_csv(log_file_all, log_file_best, gen, pop, fvals, np.argmin(fvals))
            gen += 1

    except StopIteration:
        pass

    result = solver._result()
    print("Bestes x:", result.x)
    print("Bestes f:", result.fun)
    pass

if __name__=='__main__':
    pass
