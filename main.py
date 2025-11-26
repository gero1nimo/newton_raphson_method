import matplotlib
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
    for i in range(max_iter):
        result_of_f, result_of_diff = f.subs(x,x0), df.subs(x,x0)
        if result_of_diff == 0:
            return None
        
        x_new = x0 - (result_of_f/result_of_diff)
        if abs(x_new - x0) < tolerance:
            return x_new
        x0 = x_new
    
    return None

print(newton_formula(f1, df1, 2, 1, 5))
