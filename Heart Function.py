import numpy as np
import matplotlib.pyplot as plt

# Heart equation
def heartFunc(X,Y):
    F = (X**2 + Y**2 - 1)**3 - X**2 * Y**3
    return F

# Defined the grid of points
def init():
    x = np.linspace(-1.5, 1.5, 250)
    y = np.linspace(-1.5, 1.5, 250)
    X, Y = np.meshgrid(x, y)
    return X,Y

def plot(X,Y,F):
    # Plot the contour where F = 0
    plt.contour(X, Y, F, levels=[0], colors='red')
    plt.title("Heart Shape using Cartesian Equation", fontsize=12)  
    plt.show()
    
if __name__ == "__main__":
    X, Y = init()
    F = heartFunc(X,Y)
    plot(X,Y,F)
    
