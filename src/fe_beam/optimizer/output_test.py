import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

def eggholder(x):

    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))

            -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

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

if __name__=='__main__':
    bounds = [(-1000, 1000), (-1000, 1000)]
    solver = DifferentialEvolutionSolver(
        func=eggholder,
        bounds=bounds,
        popsize=10,
        maxiter=200,
        workers=-1,
        updating='deferred'
    )
    gen = 0

    log_file_all = '/home/alex/Projects/optimizer/test_results/test_all.csv'
    log_file_best = '/home/alex/Projects/optimizer/test_results/test_best.csv'
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

            pop = solver.population
            plt.plot(pop[:, 0], pop[:, 1], 'o', ls='', label=f'gen {gen}')
            fvals = solver.population_energies

            log_population_csv(log_file_all, log_file_best, gen, pop, fvals, np.argmin(fvals))
            gen += 1

    except StopIteration:
        pass

    result = solver._result()
    print("Bestes x:", result.x)
    print("Bestes f:", result.fun)

    #best_results = np.loadtxt(log_file_best, skiprows=1, delimiter=',')
    #plt.plot(best_results[:, 0], best_results[:, 1])
    plt.legend()
    plt.show()