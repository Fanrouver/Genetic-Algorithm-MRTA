import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Problem parameters
grid_size = (20, 20)
starting_positions = {
    "Agent1": (0, 0),
    "Agent2": (0, 9),
    "Agent3": (19, 19)
}
objectives = {
    "Agent1": (9, 10),
    "Agent2": (13, 6),
    "Agent3": (6, 6)
}
obstacles = [(3, 3), (3, 4), (4, 3), (4, 4), (15, 15), (15, 16), (16, 15), (16, 16), 
             (7, 7), (7, 8), (8, 7), (8, 8), (13, 13), (13, 14), (14, 13), (14, 14),
             (11, 11), (11, 12), (12, 11), (12, 13), (6,12), (5,12), (4,12), (3,12)]

# Genetic algorithm parameters
population_size = 20
generations = 1000
time_step = 1
mutation_rate = 0.1

# Fitness function
def fitness_function(chromosome):
    total_distance = 0
    for agent, path in chromosome.items():
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            total_distance += abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1])

        # Check if the agent has reached its assigned objective
        if path[-1] == objectives[agent]:
            total_distance -= 100  # Add a bonus for reaching the objective
        else:
            total_distance += 0  # Add a penalty for not reaching the objective
    return -total_distance  # Return the negative of the total distance


# Update agent positions asynchronously
def update_positions(chromosome, time_step):
    updated_chromosome = {}
    for agent, path in chromosome.items():
        if len(path) > 1:
            current_pos = path[0]
            next_pos = path[1]
            distance = abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1])
            if distance <= time_step:
                updated_chromosome[agent] = path[1:]
            else:
                x_diff = next_pos[0] - current_pos[0]
                y_diff = next_pos[1] - current_pos[1]
                x_step = int(np.sign(x_diff)) * time_step
                y_step = int(np.sign(y_diff)) * time_step
                updated_pos = (current_pos[0] + x_step, current_pos[1] + y_step)

                # Check if the updated position is valid (not an obstacle)
                if updated_pos not in obstacles:
                    updated_chromosome[agent] = [updated_pos] + path[1:]
                else:
                    updated_chromosome[agent] = path
        else:
            updated_chromosome[agent] = path
    return updated_chromosome


# Selection function (tournament selection)
def selection(population):
    tournament_size = 3
    tournament_contestants = random.sample(population, tournament_size)
    winner = max(tournament_contestants, key=lambda chromosome: fitness_function(chromosome))
    return winner

# Crossover function (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    offspring = {}
    for agent in parent1.keys():
        offspring[agent] = parent1[agent][:crossover_point] + parent2[agent][crossover_point:]
    return offspring

# Mutation function (random position mutation)
def mutation(chromosome):
    mutated_chromosome = chromosome.copy()
    for agent in mutated_chromosome.keys():
        if random.random() < mutation_rate:
            index = random.randint(0, len(mutated_chromosome[agent]) - 1)
            mutated_chromosome[agent][index] = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
    return mutated_chromosome

# Solve the problem using a genetic algorithm
def solve_genetic_algorithm():
    # Initialize population
    population = []
    for _ in range(population_size):
        chromosome = {}
        for agent, start_pos in starting_positions.items():
            chromosome[agent] = [start_pos]
        population.append(chromosome)

    best_solution = None
    best_fitness = float('-inf')
    best_fitnesses = []
    average_fitnesses = []

    for generation in range(generations):
        # Evaluate fitness of population
        fitness_values = [fitness_function(chromosome) for chromosome in population]
        average_fitness = np.mean(fitness_values)

        # Update best solution
        max_fitness_index = np.argmax(fitness_values)
        if fitness_values[max_fitness_index] > best_fitness:
            best_fitness = fitness_values[max_fitness_index]
            best_solution = population[max_fitness_index]

        # Generate next generation
        next_generation = []
        next_generation.append(max(population, key=lambda chromosome: fitness_function(chromosome)))  # Elitism

        while len(next_generation) < population_size:
            parent1 = selection(population)
            parent2 = selection(population)
            offspring = crossover(parent1, parent2)
            offspring = mutation(offspring)
            next_generation.append(offspring)

        population = next_generation

        # Save best and average fitness for plotting
        best_fitnesses.append(best_fitness)
        average_fitnesses.append(average_fitness)

    # Plot the fitness graph
    plt.plot(range(generations), best_fitnesses, label='Best Fitness')
    plt.plot(range(generations), average_fitnesses, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

    return best_solution

# Create a grid for plotting
grid = np.zeros(grid_size)

# Solve the problem using a genetic algorithm
best_solution = solve_genetic_algorithm()

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(-1, grid_size[0])
ax.set_ylim(-1, grid_size[1])
ax.set_aspect('equal')
plt.xticks(np.arange(-0.5, grid_size[0]), [])
plt.yticks(np.arange(-0.5, grid_size[1]), [])

# Plot obstacles
for obstacle in obstacles:
    ax.add_patch(plt.Rectangle(obstacle, 1, 1, color='red'))

# Initialize the agent plots
agent_plots = {}
for agent, start_pos in starting_positions.items():
    agent_plots[agent] = ax.plot([], [], marker='o', markersize=10, label=agent)[0]
    agent_plots[agent].set_data([start_pos[0]], [start_pos[1]])

# Plot the objectives
for agent, objective in objectives.items():
    ax.plot(objective[0], objective[1], marker='*', markersize=10, color='green')

# Update function for animation
def update(frame):
    updated_positions = update_positions(best_solution, time_step)
    for agent, path in updated_positions.items():
        agent_plots[agent].set_data([pos[0] for pos in path[:frame+1]], [pos[1] for pos in path[:frame+1]])
    return list(agent_plots.values())

# Animate the agent movements
ani = animation.FuncAnimation(fig, update, frames=len(best_solution["Agent1"]), interval=1000, blit=True)
plt.legend()

plt.show()

# Print the final paths
print("\nAgent Paths:")
for agent, path in best_solution.items():
    print(f"{agent}: {path}")
