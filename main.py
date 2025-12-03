from matplotlib import pyplot as plt
import numpy as np
import sympy

x = sympy.Symbol("x")
def draw_graph(f,result):
    x_vals = np.linspace(-50, 50,1000)
    y_vals = [f.subs(x, val) for val in x_vals]
    plt.figure(num=0, dpi=100)
    plt.plot(x_vals, y_vals, color='blue')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.title(f'Plot of {f}')
    plt.xlabel(f"x values in order: {[i for i in result[1]]}")
    plt.ylabel(f"converged x value: {result[0]}")
    plt.plot(result[1], [f.subs(x, val) for val in result[1]],ls='--', marker='o', color='red')
    plt.show()

def newton_formula(f, df, x0, tolerance, max_iter):
    iteration_values = []
    x_values = []
    for i in range(max_iter):
        
        if type(x0) == complex:
            result_of_f, result_of_diff = complex(f.subs(x,x0)), complex(df.subs(x,x0))
        else:
            result_of_f, result_of_diff = float(f.subs(x,x0)), float(df.subs(x,x0))
            
        if result_of_diff == 0:
            print("Derivative is zero. No solution found.")
            break
        
        iteration_values.append({"iteration": i+1, "x_value": x0, "f(x)": result_of_f, "f'(x)": result_of_diff})
        x_values.append(x0)
        x_new = x0 - (result_of_f/result_of_diff)

        x_values.append(x_new)
        if abs(x_new - x0) < tolerance:
            iteration_values.append({"iteration": i+2, "x_value": x_new, "f(x)": f.subs(x,x_new), "f'(x)": df.subs(x,x_new)})
            return x_new, x_values, iteration_values
        
        if i == max_iter - 1:
            print("Maximum iterations reached. No solution found.")
            return "None", x_values, iteration_values
        
        x0 = x_new
        
    return None, x_values, iteration_values

f1 = (x+1)*(x-4)
df1 = sympy.diff(f1,x)
f2 = (x-1)*(x+3)
df2 = sympy.diff(f2,x)
f3 = (x-4)*(x-1)*(x+3)
df3 = sympy.diff(f3,x)
f4 = x**3 - 1
df4 = sympy.diff(f4,x) 
