## The Process of Developing this code is in github repository: https://github.com/gero1nimo/newton_raphson_method

from matplotlib import pyplot as plt
import numpy as np
import sympy

x = sympy.Symbol("x")

def draw_complex_graph(f,result, roots):
    x_vals = np.linspace(-10, 10,1000)
    y_vals = [f.subs(x, val) for val in x_vals]
    plt.figure(num=0, dpi=100)
    plt.plot(x_vals, y_vals, color='blue')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.title(f'Plot of {f} (Real Part)')
    plt.xlabel(f"Real part of x values in order: {[i.real for i in result[1]]}")
    plt.ylabel(f"Real part of converged x value: {result[0].real}")
    plt.plot([val.real for val in result[1]], [f.subs(x, val).as_real_imag()[0] for val in result[1]],ls='--', marker='o', color='red')
    plt.show()
    
def draw_graph(f,result, roots):
    
    plt.figure(figsize=(10,6), dpi=100)
    
    ## Functions Graph Values
    x_vals = np.linspace(-10, 10,1000)
    y_vals = [f.subs(x, val) for val in x_vals]
    plt.plot(x_vals, y_vals, color='blue')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    
    ## Iterations 
    x_guesses = np.array(result[1])
    y_values = np.array([f.subs(x, val) for val in x_guesses])
    plt.plot(x_guesses, y_values, color='red', marker='o', label='Iteration Points')
    for xv, yv in zip(x_guesses, y_values):
        plt.plot([xv, xv], [0, yv], color='green', lw=1)
        
    plt.plot(x_guesses, y_values,ls='--', marker='o', color='red')
    
    plt.title(f'Plot of {f}')
    plt.xlabel(f"x values in order: {[i for i in result[1]]}")
    plt.ylabel(f"converged x value: {result[0]}")
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
            return False, x_values, iteration_values
        
        x0 = x_new
        
    return False, x_values, iteration_values

f1 = (x+1)*(x-4)
f2 = (x-1)*(x+3)
f3 = (x-4)*(x-1)*(x+3)
f4 = x**3 - 1
_functions = [(f1, sympy.diff(f1,x,), [-1,4]), (f2, sympy.diff(f2,x), [1,-3]), (f3, sympy.diff(f3,x),[-3,1,4]), (f4, sympy.diff(f4,x),[1+0j, -0.5+0.8660254j , -0.5-0.8660254j])]
def newton_raphson_according_to_user_input():
    
    for f, df, roots in _functions:
        print(f"Function: {f}")
        try:
            x0 = float(input("Enter initial guess (x0): "))
        except ValueError:
            try:
                x0 = complex(input("Enter initial guess (x0): "))
            except ValueError:
                print("Invalid input for initial guess. Skipping to next function.")
                
        tolerance = float(input("Enter tolerance level: "))
        max_iter = int(input("Enter maximum number of iterations: "))
        
        result = newton_formula(f, df, x0, tolerance, max_iter)
        
        if result[0] != False:
            print(f"Converged to root: {result[0]}")
            for iter_info in result[2]:
                print(iter_info)
            if type(x0) == complex:
                draw_complex_graph(f, result, roots)
            else:
                draw_graph(f, result, roots)
        else:
            print("No root found within the given parameters.")
        
        print("-" * 40)
        

def basin_of_attraction(f,df, roots):
    print("Function: ",f)
    f_numpy = sympy.lambdify(x, f, 'numpy')
    df_numpy = sympy.lambdify(x, df, 'numpy')
    width, height = 800, 800
    min_real, max_real = -10, 10
    min_imag, max_imag = -10, 10
    
    real_values = np.linspace(min_real, max_real, width)
    imag_values = np.linspace(min_imag, max_imag, height)
    X, Y = np.meshgrid(real_values, imag_values)
    z = X + 1j * Y
    max_iterations = 50
    tolerance = 1e-6
    for i in range(max_iterations):
        f_z = f_numpy(z)
        df_z = df_numpy(z)
        df_z[df_z == 0] = 1e-12
        z = z - f_z / df_z
    
    img_ = np.zeros(z.shape, dtype=int)
    roots = np.array(roots, dtype=complex)
    
    for i, root in enumerate(roots):
        img_ += i * (np.abs(z - root) < tolerance)
   
    plt.figure(figsize=(8,8), dpi=100)
    plt.imshow(img_, extent=(min_real, max_real, min_imag, max_imag), cmap='tab10', alpha=0.9)
    real_roots = [np.real(root) for root in roots]     
    imaginary_roots = [np.imag(root) for root in roots]
    plt.scatter(real_roots, imaginary_roots, color='white', edgecolors='black', marker='x', s=100, label='Roots')    
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f)
    plt.legend()
    plt.show()

for f, df, roots in _functions:
    basin_of_attraction(f, df, roots)

# If you want to run the user input based Newton-Raphson method, call the function below    
# newton_raphson_according_to_user_input()
