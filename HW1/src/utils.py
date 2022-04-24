import matplotlib.pyplot as plt
from matplotlib import cm
from tests.examples import *

def function_values(dict0, dict1, title=None):
    function_dict_list = [dict0, dict1]
    plt.figure()
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

def function_counter_lines_2D(f, dict0, dict1, title=None):
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
    plt.figure(figsize=(12, 7))
    plt.plot(x1_dict0, x2_dict0, label='Gradient descent')
    plt.plot(x1_dict1, x2_dict1, label='Newton descent')
    plt.contour(x_co, y_co, z_co, 80)
    plt.xlabel('X values', fontsize=14)
    plt.ylabel('Y values', fontsize=14)
    plt.title(f'Gradient descent vs Newton descent with wolfe condition, {title} function', fontsize=15)
    plt.legend()
    plt.show()

def function_counter_lines_3D(f, dict0, dict1, title=None):
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
    surf = ax.plot_surface(x_co, y_co, z_co, cmap=cm.pink, alpha=0.7, linewidth=0, antialiased=False)
    ax.set_zlim(math.floor(z_co.min()), math.ceil(z.max()))
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.xlabel('X values', fontsize=14)
    plt.ylabel('Y values', fontsize=14)
    plt.title(f'Gradient descent vs Newton descent with wolfe condition, {title} function', fontsize=15)
    plt.legend()
    plt.show()