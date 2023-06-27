import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Define the problem parameters
grid_size = 10  # Size of the grid
num_agents = 3  # Number of agents
num_objectives = 3  # Number of objectives
num_obstacles = 10  # Number of obstacles

# Define the genetic algorithm parameters
population_size = 50  # Number of individuals in each generation
mutation_rate = 0.1  # Probability of mutation

# Create the main window
window = tk.Tk()
window.title("Multi-Agent Multi-Objective Pathfinding")

# Define variables for the user inputs
num_runs = tk.IntVar()
num_iterations = tk.IntVar()

# Define a list to store the best fitness values for each run
best_fitness_values = []
average_fitness_values = []

# Function to start the genetic algorithm
def start_algorithm():
    # Retrieve the user inputs
    runs = num_runs.get()
    iterations = num_iterations.get()

    # Validate the inputs
    if runs <= 0 or iterations <= 0:
        messagebox.showerror("Error", "Number of runs and iterations should be greater than zero.")
        return

    # Run the genetic algorithm for the specified number of runs and iterations
    for run in range(runs):
        # Define the grid
        grid = np.zeros((grid_size, grid_size))

        # Define the objectives
        objectives = np.random.randint(1, grid_size, size=(num_objectives, 2))

        # Define the agents and their objectives
        agents = np.random.randint(0, grid_size, size=(num_agents, 2))
        agent_objectives = np.random.randint(0, num_objectives, size=num_agents)

        # Define the obstacles
        obstacles = np.random.randint(0, grid_size, size=(num_obstacles, 2))

        # Create the figure and axis for visualization
        fig, ax = plt.subplots()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')

        # Genetic algorithm main loop
        population = np.random.randint(0, grid_size, size=(population_size, num_agents * 2))

        best_fitness_values_run = []
        average_fitness_values_run = []

        for generation in range(iterations):
            # Evaluate fitness
            fitness_scores = np.zeros(population_size)
            for i in range(population_size):
                fitness_scores[i] = np.sum(evaluate_fitness(population[i], agent_objectives, objectives))

            # Selection
            parents_indices = np.random.choice(range(population_size), size=population_size,
                                               p=fitness_scores / np.sum(fitness_scores))
            parents = population[parents_indices]

            # Create next generation
            next_generation = []
            for i in range(population_size // 2):
                parent1 = parents[i]
                parent2 = parents[population_size - 1 - i]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                next_generation.append(mutate(child1))
                next_generation.append(mutate(child2))

            population = np.array(next_generation)

            # Visualization
            ax.clear()
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect('equal')

            # Plot grid
            ax.grid(True, color='gray')

            # Plot objectives
            for i in range(num_objectives):
                objective_pos = tuple(objectives[i])
                ax.scatter(objective_pos[0], objective_pos[1], marker='*', color='red', label=f'Objective {i + 1}')

            # Plot agents and their objectives
            for i in range(num_agents):
                agent_pos = tuple(agents[i])
                objective_index = agent_objectives[i]
                ax.scatter(agent_pos[0], agent_pos[1], marker='o', color='blue', label=f'Agent {i + 1}')
                ax.annotate(f'{i + 1}', agent_pos, textcoords="offset points", xytext=(0, 10), ha='center')

                objective_pos = tuple(objectives[objective_index])
                ax.annotate(f'{objective_index + 1}', objective_pos, textcoords="offset points", xytext=(0, 10), ha='center')

            # Plot obstacles
            ax.scatter(obstacles[:, 0], obstacles[:, 1], marker='s', color='black', label='Obstacles')

            plt.title(f'Generation {generation + 1}')
            plt.legend()
            plt.pause(0.1)

            # Update agent positions asynchronously
            for i in range(num_agents):
                agent_pos = tuple(agents[i])
                objective_index = agent_objectives[i]
                new_agent_pos = move_agent(agent_pos, objectives[objective_index])
                agents[i] = new_agent_pos

            # Calculate and store fitness values for statistics
            fitness_values = []
            for individual in population:
                fitness_values.append(np.sum(evaluate_fitness(individual, agent_objectives, objectives)))

            best_fitness_values_run.append(np.min(fitness_values))
            average_fitness_values_run.append(np.mean(fitness_values))

        # Get the best solution
        best_individual = population[np.argmin(fitness_scores)]

        # Print the best solution
        print(f"Run {run + 1} - Best Solution:")
        for i in range(num_agents):
            agent_pos = tuple(agents[i])
            print(f"Agent {i + 1}: {agent_pos}")

        # Evaluate fitness of the best solution
        best_fitness = evaluate_fitness(best_individual, agent_objectives, objectives)
        best_fitness_values.append(best_fitness)
        average_fitness_values.append(np.mean(best_fitness_values_run))
        print("Best Fitness:")
        print(best_fitness)

        # Final visualization
        ax.clear()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')

        # Plot grid
        ax.grid(True, color='gray')

        # Plot objectives
        for i in range(num_objectives):
            objective_pos = tuple(objectives[i])
            ax.scatter(objective_pos[0], objective_pos[1], marker='*', color='red', label=f'Objective {i + 1}')

        # Plot agents and their objectives
        for i in range(num_agents):
            agent_pos = tuple(agents[i])
            objective_index = agent_objectives[i]
            ax.scatter(agent_pos[0], agent_pos[1], marker='o', color='blue', label=f'Agent {i + 1}')
            ax.annotate(f'{i + 1}', agent_pos, textcoords="offset points", xytext=(0, 10), ha='center')

            objective_pos = tuple(objectives[objective_index])
            ax.annotate(f'{objective_index + 1}', objective_pos, textcoords="offset points", xytext=(0, 10), ha='center')

        # Plot obstacles
        ax.scatter(obstacles[:, 0], obstacles[:, 1], marker='s', color='black', label='Obstacles')

        plt.title(f'Best Solution - Run {run + 1}')
        plt.legend()
        plt.show()

    # Generate the result graph
    generate_result_graph()

# Function to evaluate fitness
def evaluate_fitness(individual, agent_objectives, objectives):
    fitness = np.zeros(num_agents)
    for i in range(num_agents):
        agent_pos = (individual[i * 2], individual[i * 2 + 1])
        objective_pos = tuple(objectives[agent_objectives[i]])
        fitness[i] = np.abs(agent_pos[0] - objective_pos[0]) + np.abs(agent_pos[1] - objective_pos[1])
    return fitness

# Function for single-point crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, num_agents * 2)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

# Function for random mutation
def mutate(individual):
    for i in range(num_agents * 2):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(0, grid_size)
    return individual

# Function to move agent towards the objective asynchronously
def move_agent(agent_pos, objective_pos):
    new_agent_pos = list(agent_pos)
    x_diff = objective_pos[0] - agent_pos[0]
    y_diff = objective_pos[1] - agent_pos[1]

    # Determine the direction to move in
    if x_diff < 0:
        move_x = -1
    elif x_diff > 0:
        move_x = 1
    else:
        move_x = 0

    if y_diff < 0:
        move_y = -1
    elif y_diff > 0:
        move_y = 1
    else:
        move_y = 0

    # Randomly change direction with a certain probability
    if np.random.rand() < 0.2:
        if np.random.rand() < 0.5:
            move_x = np.random.choice([-1, 0, 1])
        else:
            move_y = np.random.choice([-1, 0, 1])

    # Update the agent's position
    new_agent_pos[0] += move_x
    new_agent_pos[1] += move_y

    # Ensure the agent stays within the grid
    new_agent_pos[0] = np.clip(new_agent_pos[0], 0, grid_size - 1)
    new_agent_pos[1] = np.clip(new_agent_pos[1], 0, grid_size - 1)

    return tuple(new_agent_pos)

# Function to generate result graph
def generate_result_graph():
    # Plot the best and average fitness values for each run
    x = range(1, len(best_fitness_values) + 1)
    plt.plot(x, best_fitness_values, label='Best Fitness')
    plt.plot(x, average_fitness_values, label='Average Fitness')
    plt.title('Fitness Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

# Create GUI elements
label_runs = tk.Label(window, text="Number of Runs:")
label_runs.pack()
entry_runs = tk.Entry(window, textvariable=num_runs)
entry_runs.pack()

label_iterations = tk.Label(window, text="Number of Iterations:")
label_iterations.pack()
entry_iterations = tk.Entry(window, textvariable=num_iterations)
entry_iterations.pack()

button_start = tk.Button(window, text="Start", command=start_algorithm)
button_start.pack()

# Run the GUI main loop
window.mainloop()
