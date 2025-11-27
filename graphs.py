import main
from matplotlib import pyplot as plt
import numpy as np

x_vals = np.linspace(-2, 5, 100)
y_vals = [(f1.subs(x, val)).evalf() for val in x_vals]
plt.figure(num=0, dpi=100)
plt.plot(x_vals, y_vals)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.title('Plot of f1(x) = (x+1)(x-4)')
plt.show()
f1 = (main.x+1)*(main.x-4)
df1 = main.sympy.diff(f1,main.x)
main.newton_formula(f1, df1, 2, 1, 5)
