import numpy as np
import sympy as sp
from SumOfSquares import *
import time

"""
"phi: X0 => F G !VF" starting from the initial state X0, the system will never eventually reach the state VF.
"!phi = X0 & G F VF" starting from the initial state X0, the system will always have the possibility of reaching the state VF.
"""

def generate_polynomial(variables, lower_bounds, upper_bounds):
    '''
    =========================================
    Create the respective polynomials based on the bounds (g in the paper)
    =========================================
    variables = sympy variables
    lower_bounds = numpy array of lower bound values
    upper_bounds = numpy array of upper bound values
    '''
    polynomial = [(var - lower) * (upper - var) for var, lower, upper in zip(variables, lower_bounds, upper_bounds)]
    return polynomial

dim = 2
b_degree = 2
l_degree = 2
x = sp.symbols(f'x1:{dim + 1}')
tau = 5  # discretise param
alpha = 5 * 1e-2  # heat exchange
alpha_e1 = 5 * 1e-3  # heat exchange 1
alpha_e2 = 8 * 1e-3  # heat exchange 2
temp_e = 15  # external temp
alpha_h = 3.6 * 1e-3  # heat exchange room-heater
temp_h = 55  # boiler temp
def dyf(xx):
    alpha = 0.004
    theta = 0.01
    Te = 0
    Th = 40
    mu = 0.15

    def u(x_curr):
        return 0.59 - 0.011 * x_curr

    x1_next = (1 - 2 * alpha - theta - mu * u(xx[0])) * xx[0] + xx[1] * alpha + mu * Th * u(xx[0]) + theta * Te
    x2_next = xx[0] * alpha + (1 - 2 * alpha - theta - mu * u(xx[1])) * xx[1] + mu * Th * u(xx[1]) + theta * Te
    return x1_next, x2_next

start_time = time.time()
# Define the vector field
fx = np.array(dyf(x))
prob = SOSProblem()
bsp = poly_variable('bsp', x, b_degree)
bsp_constraint = prob.add_sos_constraint(bsp, x)


L_X0 = np.array([21, 21])
U_X0 = np.array([24, 24])
LX0 = [poly_variable('LX0'+str(i+1), x, l_degree) for i in range(len(x))]
for i in LX0:
    prob.add_sos_constraint(i, x)
gX = generate_polynomial(x, L_X0, U_X0)
LX0_times_gx_list = [L*g for L, g in zip(LX0, gX)]
first_condition = prob.add_sos_constraint(bsp - sum(LX0_times_gx_list), x)

w = sp.symbols(f'w1:{dim + 1}')
L_W = np.array([20, 20])
U_W = np.array([34, 34])
LW = [poly_variable('LW'+str(i+1), w, l_degree) for i in range(len(x))]
for i in LW:
    prob.add_sos_constraint(i, w)
gW = generate_polynomial(w, L_W, U_W)
LW_times_gW_list = [L*g for L, g in zip(LW, gW)]
bsp_w = bsp.subs([(x[i], w[i]) for i in range(len(x))])
fw = dyf(w)
bsp_w_f = bsp.subs([(x[i], fw[i]) for i in range(len(x))])
bt = 0.01
second_condition = prob.add_sos_constraint(bsp_w_f - bt * bsp_w - sum(LW_times_gW_list), w)

z = sp.symbols(f'z1:{dim + 1}')
L_Z = np.array([20, 20])
U_Z = np.array([34, 34])
LZ = [poly_variable('LZ'+str(i+1), z, l_degree) for i in range(len(w))]
for i in LZ:
    prob.add_sos_constraint(i, z)
gZ = generate_polynomial(z, L_Z, U_Z)
LZ_times_gZ_list = [L*g for L, g in zip(LZ, gZ)]
bsp_z = bsp.subs([(x[i], z[i]) for i in range(len(x))])
fz = dyf(z)
bsp_z_f = bsp.subs([(x[i], fz[i]) for i in range(len(x))])
third_condition = prob.add_sos_constraint(bsp_z - bsp_z_f - sum(LZ_times_gZ_list), z)
# 21, 24, 20, 26
# x' not in VF
z0v = prob.sym_to_var(z[0])
z1v = prob.sym_to_var(z[1])
prob.add_constraint(z0v > 26 or z0v < 20 or z1v > 26 or z1v < 20)


y = sp.symbols(f'y1:{dim + 1}')
L_Y = np.array([20, 20])
U_Y = np.array([34, 34])
gY = generate_polynomial(y, L_Y, U_Y)
LY = [poly_variable('LY'+str(i+1), y, l_degree) for i in range(len(y))]
for i in LY:
    prob.add_sos_constraint(i, y)
LY_times_gY_list = [L*g for L, g in zip(LY, gY)]
bsp_y = bsp.subs([(x[i], y[i]) for i in range(len(x))])
fy = dyf(y)
bsp_y_f = bsp_y.subs([(y[i], fy[i]) for i in range(len(y))])
epsilon = sp.symbols('epsilon')
# ev = prob.sym_to_var(epsilon)
ev = 1
# epsilon_constraint = prob.add_constraint(ev > 0)
fourth_condition = prob.add_sos_constraint(bsp_y - bsp_y_f - sum(LY_times_gY_list) - epsilon, y)
y0v = prob.sym_to_var(y[0])
y1v = prob.sym_to_var(y[1])
prob.add_constraint(y0v <= 26 and y0v >= 20 and y1v <= 26 and y1v >= 20)

solver = "mosek"
prob.solve(solver=solver)
if(len(bsp_constraint.get_sos_decomp()) > 0 and \
        len(first_condition.get_sos_decomp()) > 0 and \
        len(second_condition.get_sos_decomp()) > 0 and \
        len(third_condition.get_sos_decomp()) > 0 and \
        len(fourth_condition.get_sos_decomp()) > 0):
    print(bsp_constraint.get_sos_decomp())
    print(ev)
    print(bt)

end_time = time.time()
print(f"Time cost: {end_time - start_time}")

"""
Experimental Results on March 15, 2025:
B(x1, x2) = 3.158*(-0.004*x1 - 0.016*x2 + 1)**2 + 1.11*(-0.999*x1 + x2)**2 + 0.001*x1**2
epsilon = 0.0006844474952434377
Time cost: 1.270322322845459(s)
 
Experimental Results on March 16, 2025:
Matrix([[4.064*(-0.004*x1 - 0.005*x2 + 1)**2], [1.092*(-1.0*x1 + x2)**2], [0.001*x1**2]])
1
0.01
Time cost: 1.1923344135284424
"""