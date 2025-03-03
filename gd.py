import numpy as np
import matplotlib.pyplot as plt

# Declaring global variables for m1 and m2
m1_true = 6
m2_true = np.random.rand()

# Generating Synthetic Data with Noise
def generate_data():
    x = np.random.randn(1000, 1)
    y = m1_true * x + m2_true + np.random.normal(0, 1, (1000, 1))

    # Plotting the generated data
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Noisy Data")
    plt.plot(x, m1_true * x + m2_true, color='red', label='True Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Visualization Comparing Data with and without Noise')
    plt.show()

    return x, y

# Performing linear search over the range of m1 values to find the best fit
def linear_search(x, y):
    m1_range = np.linspace(-10, 20, 1000)
    loss_values = []
    best_m1 = None
    min_loss = float('inf')  # Start with the highest possible loss

    for m in m1_range:
        y_pred = m * x + m2_true
        loss = np.mean((y - y_pred) ** 2)
        loss_values.append(loss)

        if loss < min_loss:
            min_loss = loss
            best_m1 = m  # Assigning correct m1 value

    # Plotting figure to represent Loss Vs. m1
    plt.figure(figsize=(8, 5))
    plt.plot(m1_range, loss_values, color='blue')
    plt.axvline(x=best_m1, color='red', linestyle='dashed', label=f'Best m1 = {best_m1:.4f}')
    plt.xlabel('m1 Values')
    plt.ylabel('Loss')
    plt.title('Loss vs. m1 (Linear Search)')
    plt.legend()
    plt.show()

# Performing Gradient Descent
def gradient_descent(x, y, lr=0.01, epochs=501):
    m = np.random.rand()  # Initialize m randomly
    loss_history = []
    m_history = []
    N = x.shape[0]

    for epoch in range(epochs):
        # Correcting the gradient calculation
        g_m = (-2 / N) * np.sum(x * (y - (m * x + m2_true)))
        m -= lr * g_m  # Updating m for each epoch

        loss = np.mean((y - (m * x + m2_true)) ** 2)
        loss_history.append(loss)
        m_history.append(m)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Loss = {loss:.4f}, m = {m:.4f}')

    # Plotting loss reduction over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(m_history, loss_history, color='red')
    plt.xlabel('m1')
    plt.ylabel('Loss')
    plt.title('Loss Over m1 values')
    plt.show()

    return m

def main():
    x, y = generate_data()
    linear_search(x, y)
    m1_final = gradient_descent(x, y)
    print(f'Final Optimized m1: {m1_final:.4f}')

if __name__ == "__main__":
    main()
