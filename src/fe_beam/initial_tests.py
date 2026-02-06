import numpy as np
from fe_beam.utils.helpers import BeamProperties, NXbeam
from fe_beam.core.mesh import Node, ElementConnectivity
from fe_beam.optimizer.objectives import Constraints, UserInput, FatigueTestSingle
from fe_beam.optimizer.optimizer import run_optimzation


if __name__=='__main__':
    apdl_file = r'/home/alex/Projects/FatigueTestOptimizer/src/fatiguetestoptimizer/assets/Input_Examples/NR87p5/NR87p5_S21_L75_woFinish.apdl'
    beam_properties = BeamProperties(apdl_file)
    nx_beam = NXbeam(beam_properties)
    constraints = Constraints()
    user_input = UserInput()

    bounds = [(20., 40.), #L_F
              (0, 100E3), #F
              (0., 20), #mF 
              (15., 32.), #L1 
              (0., 20), #m1 
              (32., 75.), #L2 
              (0 , 20), #m2 
              ]
    mass_limits_x = [[0, 25], [25, 40], [40, 45], [45, 50], [50, 55], [55, 60], [60, 100]]
    mass_limits_m = [[0, 20E3], [0, 10E3], [0, 5E3], [0, 3.E3], [0, 2.E3], [0, 1.E3], [0, 0.5E3]]
    masses_at_lengths = [mass_limits_x, mass_limits_m]
    constraints.masses_at_lengths = masses_at_lengths
    objective = FatigueTestSingle(nx_beam, user_input, constraints)
    out_dir = r'/home/alex/Projects/tests'
    #test_input = {'L_F': 40, 'Fampl': 5E3, 'mF': 1E3, 'L1': 15, 'm1': 500, 'L2': 60, 'm2': 100}
    #objective(test_input.values())
    run_optimzation(objective, bounds, {}, out_dir)

    

    pass