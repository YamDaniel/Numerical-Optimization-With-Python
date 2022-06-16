import unittest
from src.utils import *
from tests.examples import *
from src.constrained_min import Minimizer_function_constrained

class Minimization_constrained_function(unittest.TestCase):

    def test_qp(self):
        qp_func = Minimizer_function_constrained()
        # Test
        self.assertTrue(qp_func.interior_pt(f_0=qp, x0=np.array([0.1, 0.2, 0.7], dtype=float), ineq_constraints=qp_inequalities(), eq_constraints_mat=np.array([[1, 1, 1]]), eq_constraints_rhs=np.array([[1]]), max_outer_iterations=10, max_m=False)[1] < 1.55)
        newton_dict = qp_func.function_dict
        # Create three plots
        function_values(newton_dict, constrained=True, title='qp')
        function_counter_lines_3D(f=qp, lag=qp_func.lag, dict0=newton_dict, constrained=True, title='qp')

    def test_lp(self):
        lp_func = Minimizer_function_constrained()
        # Test
        self.assertTrue(lp_func.interior_pt(f_0=lp, x0=np.array([0.5, 0.75], dtype=float), ineq_constraints=lp_inequalities(), max_outer_iterations=10, max_m=True)[1] > 2.8)
        newton_dict = lp_func.function_dict
        # Create three plots
        function_values(newton_dict, constrained=True, title='lp')
        function_counter_lines_2D(f=lp, lag=lp_func.lag, dict0=newton_dict, constrained=True, title='lp')
        function_counter_lines_3D(f=lp, lag=lp_func.lag, dict0=newton_dict, constrained=True, title='lp')