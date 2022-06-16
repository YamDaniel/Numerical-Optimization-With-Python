from tests.examples import *
class Minimizer_function:

    def __init__(self, obj_tol=1e-12):
        self.function_dict = {}
        self.obj_tol = obj_tol

    def minimizer(self, f, x0, step_len=None, obj_tol=1e-12, param_tol=1e-8, max_cond=1e+10, hessian=False, wolfe_backtracking=False, title=None, print_status=True):
        f_history = []
        x_history = []
        x = np.array(x0).copy().reshape(-1, 1)  # start point
        f.hessian = hessian
        status = False
        if hessian:
            function_method = 'Newton descent'
            max_iter = f.max_iter_hessian
            f_x, g_x, h_x = f.func(x)
            x_history.append(x.copy())
            f_history.append(f_x)
            if print_status:
                print(f'Iteration number 0:\nFunction location is: {x.T}\nFunction value is: {f_x}')
                print('-' * 45)
            for i in range(1, max_iter+1):
                cond = np.linalg.cond(h_x)  # We calculate the condition number for inversion
                if cond > max_cond:
                    if print_status:
                        print('\nError, inversion condition number greater then max condition number')
                    status = False
                    break
                if wolfe_backtracking:
                    a = Minimizer_function.wolfe_condition_backtracking(f=f, x=x, hessian=hessian)
                else:
                    a = 1
                p_k = a * np.linalg.solve(h_x, -g_x)
                x += p_k

                f_x, g_x, h_x = f.func(x)

                x_history.append(x.copy())
                f_history.append(f_x)
                if len(f_history) and len(x_history) > 1:
                    function_values_diff = f_history[-2] - f_history[-1]
                    function_location_diff = np.linalg.norm(x_history[-2] - x_history[-1])
                    if abs(function_values_diff) < obj_tol or function_location_diff < param_tol:
                        if print_status:
                            print('\nProces status: Achieved numeric tolerance for successful termination')
                        x_history.pop()
                        f_history.pop()
                        status = True
                        break
                if abs(f_x) >= 1e+4:
                    if print_status:
                        print('\nError! Function value is too large')
                    x_history.pop()
                    f_history.pop()
                    status = False
                    break
                if print_status:
                    print(f'Iteration number {i}:\nFunction location is: {x.T}\nFunction value is: {f_x}')
                    print('-' * 45)

        else:   # If not hessian
            function_method = 'Gradient descent'
            max_iter = f.max_iter
            f_x, g_x = f.func(x)
            x_history.append(x)
            f_history.append(f_x)
            if print_status:
                print(f'Iteration number 0:\nFunction location is: {x.T}\nFunction value is: {f_x}')
                print('-' * 45)
            for i in range(1, max_iter+1):
                if wolfe_backtracking:
                    a = Minimizer_function.wolfe_condition_backtracking(f=f, x=x, step_len=step_len, hessian=hessian)
                else:
                    a = 1
                x = x.copy()
                x -= a*step_len*g_x
                f_x, g_x = f.func(x)
                x_history.append(x)
                f_history.append(f_x)
                if abs(f_x) >= 1e+4:
                    if print_status:
                        print('\nError! Function value is too large')
                    x_history.pop()
                    f_history.pop()
                    status = False
                    break
                if len(x_history) and len(x_history) > 1:
                    function_values_diff = f_history[-2]-f_history[-1]
                    function_location_diff = np.linalg.norm(x_history[-2]-x_history[-1])
                    if abs(function_values_diff) < obj_tol or function_location_diff < param_tol:
                        if print_status:
                            print('\nProces status: Achieved numeric tolerance for successful termination')
                        x_history.pop()
                        f_history.pop()
                        status = True
                        break
                if print_status:
                    print(f'Iteration number {i}:\nFunction location is: {x.T}\nFunction value is: {f_x}')
                    print('-'*45)
        self.function_dict = {'Minimization method': function_method, 'Function location list': np.hstack(x_history), 'Function value list': f_history}
        if print_status:
            print(f'{function_method}: {title} function\nIteration number {len(f_history)-1}: \nFinal function location is {x_history[-1].T}\nFinal function value is {f_history[-1]}\nStatus is {status}\n')
        return x_history[-1], f_history[-1], status

    @staticmethod
    def wolfe_condition_backtracking(f, x, step_len=None, a=1, c1=0.01, c2=0.5, hessian=False):
        if hessian:
            f_x_1, g_x_1, h_x_1 = f.func(x)
            p_k = np.linalg.solve(h_x_1, -g_x_1)
            while f.func(x + a * p_k)[0] > f_x_1 + c1 * a * g_x_1.T.dot(p_k):
                a *= c2
            return a
        else:
            f_x_1, g_x_1 = f.func(x)
            while f.func(x - a * step_len * g_x_1)[0] > f_x_1 + c1 * a * step_len * g_x_1.T.dot(-g_x_1):
                a *= c2
            return a