import matplotlib.pyplot as plt
from matplotlib import cm
from tests.examples import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def function_values(dict0=None, dict1=None, constrained=False, title=None):
    if not constrained:
        function_dict_list = [dict0, dict1]
        plt.figure(figsize=(12, 7))
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Function values', fontsize=14)
        plt.title(f'Gradient descent vs Newton descent with wolfe condition, {title} function', fontsize=20)
        x_len = abs(len(dict0['Function value list'])-len(dict1['Function value list'])) * 0.5
        x_len_ratio = max(len(dict0['Function value list']), len(dict1['Function value list']))/min(len(dict0['Function value list']), len(dict1['Function value list']))
        plt.scatter(x=[1], y=[dict0['Function value list'][0]], linewidth=3, color="red")
        for func_list in function_dict_list:
            plt.plot(np.arange(1, len(func_list['Function value list'])+1), func_list['Function value list'], label=func_list['Minimization method'])
        if x_len_ratio > 100:
            plt.xscale('log')
            plt.xlim(left=1)
        else:
            plt.xticks(np.arange(0, x_len + 1, 5))
            plt.xlim(left=1, right=x_len)
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(12, 7))
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Function values', fontsize=14)
        plt.title(f'Newton method with wolfe condition, constrained {title} function', fontsize=20)
        x_len = len(dict0['Function value list'])
        plt.scatter(x=[1], y=[dict0['Function value list'][0]], linewidth=3, color="red")
        plt.plot(np.arange(1, len(dict0['Function value list'])+1), dict0['Function value list'], label='Optimization path')
        plt.xticks(np.arange(0, x_len + 1, 1))
        plt.xlim(left=1, right=x_len)
        plt.legend()
        plt.show()

def function_counter_lines_2D(f, lag=None, dict0=None, dict1=None, constrained=False, title=None):

    if not constrained:
        function_dict_list = [dict0, dict1]
        location_list = []
        # Create range for x and y
        for dic in function_dict_list:
            for col in range(dic['Function location list'].shape[1]):
                location_list.append(np.linalg.norm(dic['Function location list'][:, col]))
        max_norm = max(location_list)
        x_y = np.linspace(-max_norm, max_norm, 40)
        # Create contour lines
        x_co, y_co = np.meshgrid(x_y, x_y)
        z = f.func(np.vstack([np.array(x_co).reshape(-1), np.array(y_co).reshape(-1)]))
        z_co = z.reshape(x_co.shape)
        x1_dict0, x2_dict0, z_dict0 = dict0['Function location list'][0], dict0['Function location list'][1],  dict0['Function value list']
        x1_dict1, x2_dict1, z_dict1 = dict1['Function location list'][0], dict1['Function location list'][1], dict1['Function value list']
        plt.figure(figsize=(12, 7))
        plt.plot(x1_dict0, x2_dict0, label='Gradient descent')
        plt.plot(x1_dict1, x2_dict1, label='Newton descent')
        plt.contour(x_co, y_co, z_co, 80)
        plt.xlabel('X values', fontsize=14)
        plt.ylabel('Y values', fontsize=14)
        plt.title(f'Gradient descent vs Newton descent with wolfe condition, {title} function', fontsize=15)
        plt.legend()
        plt.show()
    else:
        location_list = []
        location_x = np.hstack(dict0['Function location list'])
        # Create range for x and y
        for vec in location_x.T:
            location_list.append(np.linalg.norm(vec))
        max_norm = max(location_list)
        x_y = [np.linspace(-max_norm, max_norm, 40)] * location_x.shape[0]
        # Create contour lines
        x_co, y_co = np.meshgrid(*x_y)
        z = lag.feasible_coordinates(np.vstack([np.array(x_co).reshape(1, -1), np.array(y_co).reshape(1, -1)]))
        z_co = z.reshape(x_co.shape)
        x1_dict0, x2_dict0, z_dict0 = location_x[0], location_x[1],  dict0['Function value list']
        plt.figure(figsize=(12, 7))
        plt.plot(x1_dict0, x2_dict0, label='Optimization path')
        plt.contour(x_co, y_co, z_co, 60)
        plt.xlabel('X values', fontsize=14)
        plt.ylabel('Y values', fontsize=14)
        plt.title(f'Newton method with wolfe condition, constrained {title} function', fontsize=15)
        plt.legend()
        plt.show()


def function_counter_lines_3D(f, lag=None, dict0=None, dict1=None, constrained=False, title=None):
    if not constrained:
        function_dict_list = [dict0, dict1]
        location_list = []
        # Create range for x and y
        for dic in function_dict_list:
            for col in range(dic['Function location list'].shape[1]):
                location_list.append(np.linalg.norm(dic['Function location list'][:, col]))
        max_norm = max(location_list)
        x_y = np.linspace(-max_norm, max_norm, 40)
        # Create contour lines
        x_co, y_co = np.meshgrid(x_y, x_y)
        z = f.func(np.array([np.array(x_co).reshape(-1), np.array(y_co).reshape(-1)]))
        z_co = z.reshape(x_co.shape)
        x1_dict0, x2_dict0, z_dict0 = dict0['Function location list'][0], dict0['Function location list'][1],  dict0['Function value list']
        x1_dict1, x2_dict1, z_dict1 = dict1['Function location list'][0], dict1['Function location list'][1], dict1['Function value list']
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(projection='3d')
        ax.plot(x1_dict0, x2_dict0, z_dict0, label='Gradient descent')
        ax.plot(x1_dict1, x2_dict1, z_dict1, label='Newton descent')
        surf = ax.plot_surface(x_co, y_co, z_co, cmap=cm.pink, alpha=0.5, linewidth=0, antialiased=False)
        ax.set_zlim(math.floor(z_co.min()), math.ceil(z.max()))
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.xlabel('X values', fontsize=14)
        plt.ylabel('Y values', fontsize=14)
        plt.title(f'Gradient descent vs Newton descent with wolfe condition, {title} function', fontsize=15)
        plt.legend()
        plt.show()
    else:
        if dict0['Function location list'][0].shape[0] == 2:
            location_list = []
            location_x = np.hstack(dict0['Function location list'])
            # Create range for x and y
            for vec in location_x.T:
                location_list.append(np.linalg.norm(vec))
            max_norm = max(location_list)
            x_y = [np.linspace(-max_norm, max_norm, 40)] * location_x.shape[0]
            # Create contour lines
            x_co, y_co = np.meshgrid(*x_y)
            if dict0['Function value list'][0] > dict0['Function value list'][1]:
                d = 1
            else:
                d = -1
            z = lag.feasible_coordinates(np.vstack([np.array(x_co).reshape(1, -1), np.array(y_co).reshape(1, -1)]))
            z_co = z.reshape(x_co.shape) * d
            x1_dict0, x2_dict0, z_dict0 = location_x[0], location_x[1], dict0['Function value list']
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(projection='3d')
            ax.plot(x1_dict0, x2_dict0, z_dict0, label='Optimization path')
            surf = ax.plot_surface(x_co, y_co, z_co, cmap=cm.pink, alpha=0.5, linewidth=0, antialiased=False)
            min_z, max_z = np.nanmin(z_co), np.nanmax(z_co)
            ax.set_zlim(math.floor(min_z), math.ceil(max_z))
            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.xlabel('X values', fontsize=14)
            plt.ylabel('Y values', fontsize=14)
            plt.title(f'Newton method with wolfe condition, constrained {title} function', fontsize=15)
            plt.legend()
            plt.show()
        else:
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            ax.set_title("3D plot of " + 'qp')
            lag_ineq = lag.ineq_constraints.value_list
            grad_list = []
            for val in lag_ineq:
                grad_list.append(val[1])
            grad_list = -np.hstack(grad_list)
            ineqs = [[tuple(row) for row in grad_list]]
            poly_3d_collection = Poly3DCollection(ineqs, alpha=0.7, edgecolors="r")
            ax.add_collection3d(poly_3d_collection)
            location_list = np.hstack(dict0['Function location list'])
            x1 = location_list[0]
            x2 = location_list[1]
            x3 = location_list[2]
            ax.plot(x1, x2, x3, color='r', marker=".", linestyle="-", label='Optimization path')
            ax.set_zlabel("Z values")
            plt.xlabel("X values")
            plt.ylabel("Y values")
            plt.legend()
            plt.show()