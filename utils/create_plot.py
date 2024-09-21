import matplotlib.pyplot as plt

def create_plot(true_values, predictions, columns):
    for i, target in enumerate(columns):
        plt.figure(figsize=(20,6))
        plt.plot(true_values[:, i], label='True')
        plt.plot(predictions[:, i], label='Predicted')
        plt.title(f'{target} - True vs Predicted')
        plt.legend()
        plt.show()