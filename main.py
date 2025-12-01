import sympy

x = sympy.Symbol("X")

f1 = (x+1)*(x-4)
df1 = sympy.diff(f1,x)
f2 = (x-1)*(x+3)
f3 = (x-4)*(x-1)*(x+3)
f4 = x**3 - 1

############
# Test Cases for derivatives
# print(f4.subs(x,2))
# print(f1)
# print(sympy.diff(f1, x))
# print(f2)
# print(sympy.diff(f2, x))
# print(f3)
# print(sympy.diff(f3, x))
# print(f4)
# print(sympy.diff(f4, x))

def newton_formula(f, df, x0, tolerance, max_iter):
    iteration_values = []
    x_values = []
    for i in range(max_iter):
        result_of_f, result_of_diff = f.subs(x,x0), df.subs(x,x0)
        if result_of_diff == 0:
            return None
        iteration_values.append({"iteration": i+1, "x_value": x0, "f(x)": result_of_f, "f'(x)": result_of_diff})
        x_values.append(x0)
        x_new = x0 - float(result_of_f/result_of_diff)
        print(x_new)
        if abs(x_new - x0) < tolerance:
            iteration_values.append({"iteration": i+2, "x_value": x_new, "f(x)": f.subs(x,x_new), "f'(x)": df.subs(x,x_new)})
            x_values.append(x_new)
            return x_new, x_values, iteration_values
        x0 = x_new
        
    return None, x_values, iteration_values

result = newton_formula(f1, df1, 2, 1e-6, 20)
print(f"Converged x value: {float(result[0])}\n")
print("Real roots of f1: (-1, 4)\n")
print(f"All x values in order: {result[1]}\n")
print("Iterations Detail:")
for item in result[2]:
    print(item)
    
from matplotlib import pyplot as plt
import numpy as np

x_vals = np.linspace(-10, 10,100)
y_vals = [f1.subs(x, val) for val in x_vals]
plt.figure(num=0, dpi=100)
plt.plot(x_vals, y_vals, color='blue')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.title('Plot of f1(x) = (x+1)(x-4)')
plt.xlabel(f"x values in order: {[i for i in result[1]]}")
plt.ylabel(f"converged x value: {result[0]}")
plt.plot(result[1], [f1.subs(x, val) for val in result[1]],ls='--', marker='o', color='red')
plt.show()

