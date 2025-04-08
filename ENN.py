from joblib import Parallel, delayed
import numpy as np
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Cargar y preprocesar los datos
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # Normalizar entre 0 y 1
X_test = X_test.reshape(X_test.shape[0], -1) / 255

def init_params():
    """
    Inicializa los parámetros de la red neuronal con biases aleatorios pequeños.

    Returns:
        tuple: Pesos y sesgos inicializados (W1, b1, W2, b2).
    """
    W1 = np.random.randn(35, 784) * np.sqrt(2 / 784)  # He initialization para ReLU
    W2 = np.random.randn(10, 35) * np.sqrt(1 / 35)  # Xavier initialization para softmax
    b1 = np.random.randn(35, 1) * 0.01  # Biases aleatorios pequeños
    b2 = np.random.randn(10, 1) * 0.01  # Biases aleatorios pequeños
    return W1, b1, W2, b2

def ReLU(Z):
    """
    Aplica la función de activación ReLU.

    Args:
        Z (ndarray): Entrada a la función de activación.

    Returns:
        ndarray: Salida activada.
    """
    return np.maximum(0, Z)

def softmax(Z):
    """
    Aplica la función de activación softmax.

    Args:
        Z (ndarray): Entrada a la función de activación.

    Returns:
        ndarray: Probabilidades normalizadas.
    """
    exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Estabilización numérica
    return exp / np.sum(exp, axis=0, keepdims=True)

def cross_entropy(A2, Y, weights=None, lambda_reg=0.0001):
    """
    Calcula la pérdida de entropía cruzada con regularización L2 opcional.

    Args:
        A2 (ndarray): Salida de la red neuronal (predicciones).
        Y (ndarray): Etiquetas verdaderas en formato one-hot.
        weights (list): Lista de matrices de pesos (W1, W2).
        lambda_reg (float): Coeficiente de regularización L2.

    Returns:
        float: Pérdida promedio con regularización.
    """
    # Transponer Y para que coincida con la forma de A2
    Y = Y.T
    loss = -np.mean(np.sum(Y * np.log(A2 + 1e-8), axis=0))  # Evitar log(0) añadiendo 1e-8
    if weights is not None:
        l2_penalty = lambda_reg * sum(np.sum(W**2) for W in weights)
        loss += l2_penalty
    return loss

def forward_prop(W1, b1, W2, b2, X):
    """
    Realiza la propagación hacia adelante en la red neuronal.

    Args:
        W1, b1, W2, b2 (ndarray): Pesos y sesgos de la red.
        X (ndarray): Datos de entrada.

    Returns:
        ndarray: Salida de la red neuronal.
    """
    Z1 = W1.dot(X.T) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2

def one_hot(Y):
    """
    Convierte etiquetas en formato one-hot.

    Args:
        Y (ndarray): Etiquetas originales.

    Returns:
        ndarray: Etiquetas en formato one-hot.
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

def flatten_params(W1, b1, W2, b2):
    """
    Aplana los parámetros de la red neuronal en un solo vector.

    Args:
        W1, b1, W2, b2 (ndarray): Pesos y sesgos de la red.

    Returns:
        ndarray: Vector de parámetros aplanados.
    """
    return np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

def reconstruct_params(vec):
    """
    Reconstruye los parámetros de la red neuronal desde un vector aplanado.

    Args:
        vec (ndarray): Vector de parámetros aplanados.

    Returns:
        tuple: Pesos y sesgos reconstruidos (W1, b1, W2, b2).
    """
    W1 = vec[:35*784].reshape(35, 784)  # 35 * 784
    b1 = vec[35*784:35*784+35].reshape(35, 1)  # 35
    W2 = vec[35*784+35:35*784+35+10*35].reshape(10, 35)  # 10 * 35
    b2 = vec[35*784+35+10*35:].reshape(10, 1)  # 10
    return (W1, b1, W2, b2)

def generate_population(pop_size):
    """
    Genera una población inicial de individuos.

    Args:
        pop_size (int): Tamaño de la población.

    Returns:
        list: Lista de vectores de parámetros aplanados.
    """
    population = []
    for _ in range(pop_size):
        W1, b1, W2, b2 = init_params()
        individual = flatten_params(W1, b1, W2, b2)
        population.append(individual)
    return population

def select_population(population, fitness_scores, num_selected):
    """
    Selecciona una nueva población basada en la pérdida, priorizando a los individuos con menor pérdida
    pero conservando diversidad.

    Args:
        population (list): Lista de individuos (vectores de parámetros aplanados).
        fitness_scores (list): Lista de pérdidas correspondientes a cada individuo.
        num_selected (int): Número de individuos a seleccionar.

    Returns:
        list: Nueva población seleccionada.
    """
    # Emparejar individuos con sus pérdidas
    paired_population = list(zip(population, fitness_scores))
    max_loss = max(fitness_scores)
    probabilities = [(max_loss - score) for _, score in paired_population]  # Invertir la escala
    total_probability = sum(probabilities)
    probabilities = [p / total_probability for p in probabilities]  # Normalizar

    selected_population = []
    for _ in range(num_selected):
        tournament = random.choices(paired_population, weights=probabilities, k=3)
        winner = min(tournament, key=lambda x: x[1])  # Seleccionar el de menor pérdida
        selected_population.append(winner[0])  # Agregar solo el individuo

    return selected_population

def crossover(parent1, parent2, parent3):
    """
    Genera un nuevo individuo combinando segmentos de tres padres directamente en forma de vector.
    Asegura que los genes intercambiados correspondan a las regiones de pesos y biases.

    Args:
        parent1, parent2, parent3 (ndarray): Vectores de parámetros aplanados de los padres.

    Returns:
        ndarray: Vector de parámetros aplanados del hijo.
    """
    w1_size = 35 * 784  # Tamaño de W1
    b1_size = 35        # Tamaño de b1
    w2_size = 10 * 35   # Tamaño de W2
    b2_size = 10        # Tamaño de b2

    child = np.zeros_like(parent1)

    block_size = 10  # Tamaño del bloque
    for i in range(0, w1_size, block_size):
        choice = random.choice([1, 2, 3])
        if choice == 1:
            child[i:i+block_size] = parent1[i:i+block_size]
        elif choice == 2:
            child[i:i+block_size] = parent2[i:i+block_size]
        else:
            child[i:i+block_size] = parent3[i:i+block_size]

    for i in range(w1_size, w1_size + b1_size):
        choice = random.choice([1, 2, 3])
        child[i] = parent1[i] if choice == 1 else parent2[i] if choice == 2 else parent3[i]

    for i in range(w1_size + b1_size, w1_size + b1_size + w2_size):
        choice = random.choice([1, 2, 3])
        child[i] = parent1[i] if choice == 1 else parent2[i] if choice == 2 else parent3[i]

    for i in range(w1_size + b1_size + w2_size, w1_size + b1_size + w2_size + b2_size):
        choice = random.choice([1, 2, 3])
        child[i] = parent1[i] if choice == 1 else parent2[i] if choice == 2 else parent3[i]

    return child

def fitness_scores(population, X, Y):
    """
    Calcula las puntuaciones de aptitud para cada individuo en la población de manera paralela.

    Args:
        population (list): Lista de vectores de parámetros aplanados.
        X (ndarray): Datos de entrada.
        Y (ndarray): Etiquetas verdaderas en formato one-hot.

    Returns:
        list: Lista de pérdidas (menor pérdida = mayor aptitud).
    """
    def evaluate_individual(individual):
        W1, b1, W2, b2 = reconstruct_params(individual)
        A2 = forward_prop(W1, b1, W2, b2, X)
        return cross_entropy(A2, Y, weights=[W1, W2])  # Agregar regularización L2

    # Paralelizar el cálculo de pérdidas para cada individuo
    scores = Parallel(n_jobs=-1)(delayed(evaluate_individual)(ind) for ind in population)
    return scores

def mutate(individual, mutation_rate=0.01, mutation_strength=0.1):
    """
    Aplica mutación a un individuo para explorar el espacio de estados.

    Args:
        individual (ndarray): Vector de parámetros aplanados del individuo.
        mutation_rate (float): Probabilidad de mutación para cada gen.
        mutation_strength (float): Magnitud del cambio aplicado a los genes mutados.

    Returns:
        ndarray: Individuo mutado.
    """
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] += np.random.normal(0, mutation_strength)
    return mutated_individual

# Lista global para almacenar las mejores pérdidas
best_losses = []

def plot_progress(best_losses):
    """
    Grafica el progreso de la pérdida a lo largo de las generaciones.

    Args:
        best_losses (list): Lista de las mejores pérdidas por generación.
    """
    if not best_losses:
        print("Error: No hay datos en best_losses para graficar.")
        return
    plt.plot(range(1, len(best_losses) + 1), best_losses, marker='o', linestyle='-')
    plt.xlabel('Generación')
    plt.ylabel('Mejor pérdida')
    plt.title('Progreso de la pérdida')
    plt.grid(True)
    plt.show()

def evolve(population, X, Y, num_generations, num_selected, mutation_rate=0.01, mutation_strength=0.1):
    """
    Realiza el proceso de evolución sobre una población durante varias generaciones.

    Args:
        population (list): Población inicial (lista de vectores de parámetros aplanados).
        X (ndarray): Datos de entrada.
        Y (ndarray): Etiquetas verdaderas en formato one-hot.
        num_generations (int): Número de generaciones a evolucionar.
        num_selected (int): Número de individuos seleccionados para la siguiente generación.
        mutation_rate (float): Probabilidad de mutación para cada gen.
        mutation_strength (float): Magnitud del cambio aplicado a los genes mutados.

    Returns:
        list: Población final después de la evolución.
    """
    global best_losses  # Usar la lista global para acumular las pérdidas

    for generation in range(num_generations):
        # Calcular las puntuaciones de aptitud
        fitness = fitness_scores(population, X, Y)

        # Seleccionar los mejores individuos
        selected_population = select_population(population, fitness, num_selected)

        # Generar nueva población mediante cruce y mutación
        new_population = []
        for _ in range(len(population)):
            parents = random.sample(selected_population, 3)
            child = crossover(parents[0], parents[1], parents[2])
            child = mutate(child, mutation_rate, mutation_strength)
            new_population.append(child)

        # Reemplazar la población actual con la nueva
        population = new_population

        # Registrar la mejor pérdida de la generación actual
        best_fitness = min(fitness)
        best_losses.append(best_fitness)

    return population

def evaluate_population(population, X, Y):
    """
    Evalúa la población y devuelve la mejor pérdida y el índice del mejor individuo.

    Args:
        population (list): Lista de vectores de parámetros aplanados.
        X (ndarray): Datos de entrada.
        Y (ndarray): Etiquetas verdaderas en formato one-hot.

    Returns:
        tuple: Mejor pérdida, índice del mejor individuo.
    """
    fitness = fitness_scores(population, X, Y)
    best_fitness = min(fitness)
    best_index = fitness.index(best_fitness)
    return best_fitness, best_index

def visualize_predictions(individual, X, Y, num_samples=5):
    """
    Imprime ejemplos de predicciones realizadas por un individuo.

    Args:
        individual (ndarray): Vector de parámetros aplanados del individuo.
        X (ndarray): Datos de entrada.
        Y (ndarray): Etiquetas verdaderas en formato one-hot.
        num_samples (int): Número de ejemplos a visualizar.
    """
    W1, b1, W2, b2 = reconstruct_params(individual)
    A2 = forward_prop(W1, b1, W2, b2, X)  # A2 tiene forma (10, número_de_imágenes)
    predictions = np.argmax(A2, axis=0)  # Predicciones por columna
    true_labels = np.argmax(Y, axis=1)   # Etiquetas verdaderas por fila

    print("\nEjemplos de predicciones:")
    for i in range(num_samples):
        print(f"Predicción: {predictions[i]}, Etiqueta verdadera: {true_labels[i]}")

def calculate_accuracy(individual, X, Y):
    """
    Calcula la precisión del individuo en los datos proporcionados.

    Args:
        individual (ndarray): Vector de parámetros aplanados del individuo.
        X (ndarray): Datos de entrada.
        Y (ndarray): Etiquetas verdaderas en formato one-hot.

    Returns:
        float: Precisión del individuo.
    """
    W1, b1, W2, b2 = reconstruct_params(individual)
    A2 = forward_prop(W1, b1, W2, b2, X)  # Salida de la red neuronal
    predictions = np.argmax(A2, axis=0)  # Predicciones
    true_labels = np.argmax(Y, axis=1)   # Etiquetas verdaderas
    accuracy = np.mean(predictions == true_labels)  # Precisión
    return accuracy

def plot_predictions(individual, X, Y, num_samples=20):
    """
    Genera una imagen con las predicciones realizadas por un individuo.

    Args:
        individual (ndarray): Vector de parámetros aplanados del individuo.
        X (ndarray): Datos de entrada.
        Y (ndarray): Etiquetas verdaderas en formato one-hot.
        num_samples (int): Número de ejemplos a visualizar.
    """
    W1, b1, W2, b2 = reconstruct_params(individual)
    A2 = forward_prop(W1, b1, W2, b2, X)  # Salida de la red neuronal
    predictions = np.argmax(A2, axis=0)  # Predicciones
    true_labels = np.argmax(Y, axis=1)   # Etiquetas verdaderas

    # Seleccionar índices aleatorios para visualizar
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    selected_images = X[indices].reshape(-1, 28, 28)  # Restaurar forma original de las imágenes
    selected_predictions = predictions[indices]
    selected_true_labels = true_labels[indices]

    # Configurar el tamaño de la figura
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4 filas, 5 columnas
    fig.suptitle("Predicciones del mejor individuo", fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(selected_images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Pred: {selected_predictions[i]}\nTrue: {selected_true_labels[i]}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar el espacio para el título
    plt.show()

# Configuración inicial
pop_size = 200           # Tamaño de la población
num_generations = 500     # Número de generaciones
num_selected = 40        # Número de individuos seleccionados
mutation_rate = 0.01     # Tasa de mutación
mutation_strength = 0.1  # Fuerza de mutación

# Generar población inicial
population = generate_population(pop_size)

# Convertir etiquetas a formato one-hot
Y_train_one_hot = one_hot(y_train)

# Ejecutar el algoritmo evolutivo
for generation in range(num_generations):
    print(f"\n--- Generación {generation + 1} ---")

    # Evaluar la población
    best_fitness, best_index = evaluate_population(population, X_train, Y_train_one_hot)
    print(f"Mejor pérdida: {best_fitness:.4f}")

    # Visualizar predicciones del mejor individuo
    visualize_predictions(population[best_index], X_train, Y_train_one_hot, num_samples=5)

    # Evolucionar la población
    population = evolve(population, X_train, Y_train_one_hot, 1, num_selected, mutation_rate, mutation_strength)

# Graficar el progreso de la pérdida después de todas las generaciones
plot_progress(best_losses)

# Evaluar el mejor individuo en los datos de prueba
best_fitness, best_index = evaluate_population(population, X_test, one_hot(y_test))
best_individual = population[best_index]

# Calcular la exactitud en los datos de prueba
accuracy = calculate_accuracy(best_individual, X_test, one_hot(y_test))
print(f"\nExactitud del mejor individuo en los datos de prueba: {accuracy:.2%}")

# Visualizar 20 predicciones del mejor individuo en los datos de prueba
plot_predictions(best_individual, X_test, one_hot(y_test), num_samples=20)
