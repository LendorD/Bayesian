import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from bayes_opt_new import BayesianOptimization
# from bayes_opt import BayesianOptimization
np.seterr(all='warn')

# x = np.random.normal()
# print(x)
# + random.randint(1, 5)
def black_box_function(x, y, a=1, b=2, g=5):
    return ((y) ** 2) * np.sin(a * x + b * y)
    # return ((y) ** 2) * np.sin(x + y)
    # return a*x**3*y**7 + b*x**2*y**8 - 4*x*y**9 + a*x**10 + b*y**10 + a*x**9*y + 8*x**2*y**2 - b*x**5*y**5 + 10

# Bounded region of parameter space
pbounds = {'x': (-7, 7), 'y': (-7, 7), 'a' : (0,6), 'b': (0,6), 'g': (0,6)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=3,
    verbose=2,

)

optimizer.maximize(
    init_points=5,
    n_iter=20
)



def f(x, y, a=1.063, b=2.675, g=0.2):
    return ((y) ** 2) * np.sin(a * x + b * y)
    # return ((y) ** 2) * np.sin(x + y)
#     # return 2*x**3*y**7 + 3*x**2*y**8 - 4*x*y**9 + 5*x**10 + 6*y**10 + 7*x**9*y + 8*x**2*y**2 - 9*x**5*y**5 + 10
# def df_dx(x, y,a=1.063, b=2.675, g=0.2):
#     return a*(y**2) * np.cos(a*x+b*y)
#     # return (y ** 2) * np.cos(x +  y)
#     # return 50*x**9 + 63*x**8 * y - 45 *x**4 * y**5 + 6*x**2 * y**7 + 6*x * y**8 + 16*x * y**2 - 4*y**9
# def df_dy(x, y, a=1.063, b=2.675, g=0.2):
#     return b * (y**2) * np.cos(a*x+b*y) + 2*y*np.sin(a*x+b*y)
#     # return (y ** 2) * np.cos(x +  y) + 2 * y * np.sin( x +  y)
#     # return 7*x**9 - 45*x**5 * y**4 + 14*x**3* y**6 +24*x**2 * y**7 + 16*x**2 * y - 36*x* y**8 + 60 *y**9



def df(x, y, a=1.063, b=2.675):
    return -(a*y * np.cos(a*x+y*b))/(b*y*np.cos(a*x+y*b) + np.sin(a*x+b*y))

# def gradient_descent_max(f, df_dx, df_dy, x0, y0, learning_rate=0.03, max_iterations=3000, tolerance=1e-6):
#     x, y = x0, y0
#     for i in range(max_iterations):
#         grad_x = df_dx(x, y)
#         grad_y = df_dy(x, y)
#         grad = np.array([grad_x, grad_y], dtype=np.double)
#         # if np.linalg.norm(grad) < tolerance:
#         #     break
#         x += learning_rate * grad_x
#         y += learning_rate * grad_y
#     return x, y, f(x, y)


def gradient_descent(f, df, x0, y0, lr=0.03, tol=1e-6, max_iter=1000):
    x = np.array([x0, y0])
    iter_num = 0

    while iter_num < max_iter:
        grad = np.array([df(x[0], x[1]), df(x[0], x[1])])

        x_new = x - lr * grad

        # проверяем условие на достижение требуемой точности
        if np.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new
        iter_num += 1

    # если достигнуто максимальное число итераций, то возвращаем последнюю точку
    print(x[0], x[1])
    return x

x0 = -2
y0 = -2
#
# x0 = 5
# y0 = -5

f_max = gradient_descent(f, df, x0, y0)
print("f_max from Gradinet:",  f(f_max[0],f_max[1]))



#
# def df(x, y):
#     return -(y * np.cos(x+y))/(y*np.cos(x+y) + 2 * np.sin(x+y))
#
# def d2f_dx2(x, y):
#     return -(y*(y*np.sin(x+y) - 2 * np.cos(x+y))*np.cos(x+y))/(y*np.cos(x+y)+2*np.sin(x+y))**2 + (y*np.sin(x+y))/(y*np.cos(x+y) + 2*np.sin(x+y))
#
# def d2f_dy2(x, y):
#     return -(y*(y*np.sin(x+y) - 3* np.cos(x+y))*np.cos(x+y))/(y*np.cos(x+y)+2*np.sin(x+y))**2 + (y*np.sin(x+y))/(y*np.cos(x+y) + 2*np.sin(x+y)) - np.cos(x+y)/(y*np.cos(x+y) + 2 * np.sin(x+y))
#
#
# def d2f_dxdy(x, y):
#     return -y/(y - (np.sin(2*x+2*y)/2))
#
# def newton_method_max(f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy, x0, y0, max_iterations=1000, tolerance=1e-6):
#
#     x, y = x0, y0
#     for i in range(max_iterations):
#         grad_x = df_dx(x, y)
#         grad_y = df_dy(x, y)
#         hessian = np.array([[d2f_dx2(x, y), d2f_dxdy(x, y)], [d2f_dxdy(x, y), d2f_dy2(x, y)]])
#         if np.linalg.det(hessian) == 0:
#             break
#         hessian_inv = np.linalg.inv(hessian)
#         grad = np.array([grad_x, grad_y])
#         delta = -hessian_inv.dot(grad)
#         x += delta[0]
#         y += delta[1]
#         if np.linalg.norm(delta) < tolerance:
#             break
#     return x, y, f(x, y)
#
# x0, y0 = -2, 4
# # x, y, f = newton_method_max(f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy, x0, y0)
# # print('Minimum point from Newton:', x, y)
# # print('Max value from Newton:', f)
#


###################################################################################################################################


#
#
#
# f1 = [0] * 25
# f2 = [0] * 25
# for i, res in enumerate(optimizer.res):
#     f1[i] = black_box_function(res['params']['x'], res['params']['y'])
#     f2[i] = black_box_function(res['params']['x'], res['params']['y'], res['params']['a'], res['params']['b'], res['params']['g'])
#
#
#
# newF = [0]*25
#
# for i in range(25):
#     newF[i] = math.sqrt(pow(f1[i] - f2[i], 2)/25)
#
# # new_f = black_box_function(optimizer.max['params']['x'], optimizer.max['params']['y'], optimizer.max['params']['a'], optimizer.max['params']['b'], optimizer.max['params']['g'])
#
#
# fig, ax = plt.subplots()
# # Добавим заголовок графика
# ax.set_title('График функции ошибки ')
# # Название оси X:
# ax.set_xlabel('x')
# # Название оси Y:
# ax.set_ylabel('y')
# # Начало и конец изменения значения X, разбитое на 100 точек
# x = np.linspace(0, 25, 25) # X от 0 до 5
# # Построение прямой
# # Вывод графика
# ax.plot(x, newF)
# plt.show()
#
# fig1, ax1 = plt.subplots()
# # Добавим заголовок графика
# ax1.set_title('График функции')
# # Название оси X:
# ax1.set_xlabel('x')
# # Название оси Y:
# ax1.set_ylabel('y')
# # Начало и конец изменения значения X, разбитое на 100 точек
# x1 = np.linspace(0, 25, 25) # X от 0 до 5
# # Построение прямой
# # Вывод графика
# ax1.plot(x1, f1)
# plt.show()
#
# fig2, ax2 = plt.subplots()
# # Добавим заголовок графика
# ax2.set_title('График функции от рандомных параметров')
# # Название оси X:
# ax2.set_xlabel('x')
# # Название оси Y:
# ax2.set_ylabel('y')
# # Начало и конец изменения значения X, разбитое на 100 точек
# x2 = np.linspace(0, 25, 25) # X от 0 до 5
# # Построение прямой
# # Вывод графика
# ax2.plot(x2, f2)
# plt.show()
#
# print(newF)