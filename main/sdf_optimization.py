import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


def optimize_push(x_0, sdf_func, push_pos, object_pos, goal_pos):
    # our objective which is the euclidean distance between the starting pos and our to-be-optimized point
    def objective(x):
        return np.linalg.norm(x - x_0)

    def constraint1(x):  # sdf function to be 0, so we touch the object
        return sdf_func(x)

    def constraint2(x):  # so that we can assure our point is definitely pushing from the correct x-direction
        yx = x - [push_pos[0], 0, 0]
        return sdf_func(yx)

    def constraint3(x):  # so that we can assure our point is definitely pushing from the correct y-direction
        yy = x - [0, push_pos[1], 0]
        return sdf_func(yy)

    def constraint4(x):  # make sure that we are not above the object
        yz = x - [0, 0, 0.1]
        return sdf_func(yz)

    def constraint5(x):  # make sure that the object is between us
        return np.linalg.norm(x - goal_pos) - np.linalg.norm(object_pos - goal_pos)

    con1 = NonlinearConstraint(constraint1, 0, 0)
    con2 = NonlinearConstraint(constraint2, 0, np.inf)
    con3 = NonlinearConstraint(constraint3, 0, np.inf)
    con4 = NonlinearConstraint(constraint4, 0, 0)
    con5 = NonlinearConstraint(constraint5, 0, np.inf)

    cons = [con1, con2, con3, con4]

    b = (-20, 20)
    bnds = [b, b, b]
    sol = minimize(objective, x_0, method='trust-constr', bounds=bnds, constraints=cons, options={"maxiter": 1000, 'verbose':1})

    return sol.x, sol.nit, sol.execution_time


def optimize_grasp(x_0, sdf_func):
    print("fingers start:", x_0)

    # our objective which is the euclidean distance between the starting pos and our to-be-optimized point
    def objective(x):
        return (np.linalg.norm(x[:3] - x_0[:3]) + np.linalg.norm(x[-3:] - x_0[-3:])) / 10

    def constraint1(x):  # sdf function to be 0, so the first grasping sphere touches the object
        return sdf_func(x[:3])

    def constraint2(x):  # sdf function to be 0, so the second grasping sphere touches the object
        return sdf_func(x[-3:])

    def constraint3(x):  # make sure that the grasping spheres are on the same height
        return x[:3][2] - x[-3:][2]

    con1 = NonlinearConstraint(constraint1, 0, 0)
    con2 = NonlinearConstraint(constraint2, 0, 0)
    con3 = NonlinearConstraint(constraint3, 0, 0)

    cons = [con1, con2, con3]

    b = (-20, 20)
    bnds = [b, b, b, b, b, b]

    sol = minimize(objective, x_0, method='trust-constr', bounds=bnds, constraints=cons, options={"maxiter": 1000, 'verbose':1})

    print("\n")

    return sol.x, sol.nit, sol.execution_time
