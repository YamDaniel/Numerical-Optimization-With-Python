import unittest
from src.utils import *
from examples import *
from src.unconstrained_min import Minimizer_function

class Minimization_function(unittest.TestCase):

    def test_q1(self):
        q1_func = Minimizer_function()
        self.assertFalse(q1_func.minimizer(f=q1, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=False)[2])  # Check status
        self.assertFalse(q1_func.minimizer(f=q1, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=True)[2])   # Check status
        gradient_dict = q1_func.function_dict
        self.assertTrue(q1_func.minimizer(f=q1, step_len=1e-2, x0=np.array([1, 1], dtype=float), hessian=True, wolfe_backtracking=False)[1] < 1e-7)  # Check function convergence
        self.assertTrue(q1_func.minimizer(f=q1, step_len=1e-2, x0=np.array([1, 1], dtype=float), hessian=True, wolfe_backtracking=True)[1] < 1e-7)   # Check function convergence
        newton_dict = q1_func.function_dict
        # Create two plots
        function_values(gradient_dict, newton_dict, title='Q1')
        function_counter_lines_2D(f=q1, dict0=gradient_dict, dict1=newton_dict, title='Q1')
        function_counter_lines_3D(f=q1, dict0=gradient_dict, dict1=newton_dict, title='Q1')

    def test_q2(self):
        q2_func = Minimizer_function()
        self.assertFalse(q2_func.minimizer(f=q2, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=False)[2])   # Check status
        self.assertFalse(q2_func.minimizer(f=q2, x0=np.array([1, 1], dtype=float), step_len=1e-1, wolfe_backtracking=True)[2])    # Check status
        gradient_dict = q2_func.function_dict
        self.assertTrue(q2_func.minimizer(f=q2, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=False)[1] < 1e-7)   # Check function convergence
        self.assertTrue(q2_func.minimizer(f=q2, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=True)[1] < 1e-7)    # Check function convergence
        newton_dict = q2_func.function_dict
        # Create two plots
        function_values(gradient_dict, newton_dict, title='Q2')
        function_counter_lines_2D(f=q2, dict0=gradient_dict, dict1=newton_dict, title='Q2')
        function_counter_lines_3D(f=q2, dict0=gradient_dict, dict1=newton_dict, title='Q2')

    def test_q3(self):
        q3_func = Minimizer_function()
        self.assertFalse(q3_func.minimizer(f=q3, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=False)[2])   # Check status
        self.assertFalse(q3_func.minimizer(f=q3, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=True)[2])    # Check status
        gradient_dict = q3_func.function_dict
        self.assertTrue(q3_func.minimizer(f=q3, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=False)[1] < 1e-7)   # Check function convergence
        self.assertTrue(q3_func.minimizer(f=q3, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=True)[1] < 1e-7)    # Check function convergence
        newton_dict = q3_func.function_dict
        # Create two plots
        function_values(gradient_dict, newton_dict, title='Q3')
        function_counter_lines_2D(f=q3, dict0=gradient_dict, dict1=newton_dict, title='Q3')
        function_counter_lines_3D(f=q3, dict0=gradient_dict, dict1=newton_dict, title='Q3')

    def test_rosenbrock(self):
        rosenbrock_func = Minimizer_function()
        self.assertFalse(rosenbrock_func.minimizer(f=f_rosenbrock, x0=np.array([-1, 2], dtype=float), step_len=1e-2, wolfe_backtracking=False)[2])    # Check status
        self.assertTrue(rosenbrock_func.minimizer(f=f_rosenbrock,  x0=np.array([-1, 2], dtype=float), step_len=1e-2, wolfe_backtracking=True)[2])     # Check status
        gradient_dict = rosenbrock_func.function_dict
        self.assertTrue(rosenbrock_func.minimizer(f=f_rosenbrock,  x0=np.array([-1, 2], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=False)[1] < 1e-7)    # Check function convergence
        self.assertTrue(rosenbrock_func.minimizer(f=f_rosenbrock,  x0=np.array([-1, 2], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=True)[1] < 1e-7)     # Check function convergence
        newton_dict = rosenbrock_func.function_dict
        # Create two plots
        function_values(gradient_dict, newton_dict, title='Rosenbrock')
        function_counter_lines_2D(f=f_rosenbrock, dict0=gradient_dict, dict1=newton_dict, title='Rosenbrock')
        function_counter_lines_3D(f=f_rosenbrock, dict0=gradient_dict, dict1=newton_dict, title='Rosenbrock')

    def test_linear(self):
        linear_func = Minimizer_function()
        self.assertFalse(linear_func.minimizer(f=f_linear, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=False)[2])    # Check status
        self.assertFalse(linear_func.minimizer(f=f_linear, x0=np.array([1, 1], dtype=float), step_len=0.1, wolfe_backtracking=True)[2])     # Check status
        gradient_dict = linear_func.function_dict
        self.assertFalse(linear_func.minimizer(f=f_linear, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=False)[2])    # Check status
        self.assertFalse(linear_func.minimizer(f=f_linear, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=True)[2])     # Check status
        newton_dict = linear_func.function_dict
        # Create two plots
        function_values(gradient_dict, newton_dict, title='Linear')
        function_counter_lines_2D(f=f_linear, dict0=gradient_dict, dict1=newton_dict, title='Linear')
        function_counter_lines_3D(f=f_linear, dict0=gradient_dict, dict1=newton_dict, title='Linear')

    def test_exponential(self):
        exponential_func = Minimizer_function()
        self.assertFalse(exponential_func.minimizer(f=f_exponential, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=False)[2])    # Check status
        self.assertFalse(exponential_func.minimizer(f=f_exponential, x0=np.array([1, 1], dtype=float), step_len=1e-2, wolfe_backtracking=True)[2])     # Check status
        gradient_dict = exponential_func.function_dict
        self.assertTrue(exponential_func.minimizer(f=f_exponential, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=False)[1] < 2.55927 + 1e-5)     # Check function convergence
        self.assertTrue(exponential_func.minimizer(f=f_exponential, x0=np.array([1, 1], dtype=float), step_len=1e-2, hessian=True, wolfe_backtracking=True)[1] < 2.55927 + 1e-5)      # Check function convergence
        newton_dict = exponential_func.function_dict
        # Create two plots
        function_values(gradient_dict, newton_dict, title='Exponential')
        function_counter_lines_2D(f=f_exponential, dict0=gradient_dict, dict1=newton_dict, title='Exponential')
        function_counter_lines_3D(f=f_exponential, dict0=gradient_dict, dict1=newton_dict, title='Exponential')

if __name__ == '__main__':
    unittest.main()