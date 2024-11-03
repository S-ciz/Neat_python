import numpy as np
import pygame
import random
from collections import namedtuple, defaultdict
import pickle
from genome import Genome
from game import SnakeGame

GRID_SIZE = 10
CELL_SIZE = 30
POPULATION_SIZE = 50
GENERATIONS = 500
MAX_STEPS = 500  
COMPATIBILITY_THRESHOLD = 3.0
SPECIES_ELITISM = 2 
RECENT_ACTIONS_HISTORY = 5  


def save_genome(genome, filename):
    with open(filename, 'wb') as f:
        pickle.dump(genome, f)

def load_genome(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def evaluate_fitness(genome):
    game = SnakeGame()
    total_score = 0
    survival_bonus = 0

    # Initialize the action history
    recent_actions = []

    # Initial distance to food
    head = game.snake[0]
    initial_distance = np.sqrt((head.x - game.food.x) ** 2 + (head.y - game.food.y) ** 2)

    while game.steps < MAX_STEPS:
        state = game.get_state()
        action = np.argmax(genome.forward(state))
        
        # Penalize fo repeating actions
        if len(recent_actions) > 0 and action == recent_actions[-1]:
            total_score -= 0.5  # Small penalty for repeating the same action

        # Udate the action history
        recent_actions.append(action)
        if len(recent_actions) > RECENT_ACTIONS_HISTORY:
            recent_actions.pop(0)  # Keep only the last N actions

        done, score = game.play_step(action)

        if done:
            break
        
        total_score += score
        survival_bonus += 1  # Bonus for surviving longer
        
        # Calculate new distance to food
        head = game.snake[0]
        new_distance = np.sqrt((head.x - game.food.x) ** 2 + (head.y - game.food.y) ** 2)

        # Reward for moving closer to the food
        if new_distance < initial_distance:
            total_score += 1  
        else:
            total_score -= 0.5 
        
        initial_distance = new_distance  

    return total_score + 0.1 * survival_bonus


# evolutionary Algorithm with Speciation
def speciate_population(population):
    species = defaultdict(list)
    for genome in population:
        found_species = False
        for other_genome in species.keys():
            if genome.compatibility_distance(other_genome) < COMPATIBILITY_THRESHOLD:
                species[other_genome].append(genome)
                found_species = True
                break
        if not found_species:
            species[genome] = [genome]
    return species

def evolve_population(population):
    species = speciate_population(population)
    new_population = []

    for specie, genomes in species.items():
        genomes.sort(key=lambda g: g.fitness, reverse=True)  # Sort by fitness
        new_population.extend(genomes[:SPECIES_ELITISM])  # Keep top performers

        # create child:: through crossover
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(genomes[:len(genomes)//2], k=2)
            child = parent1.crossover(parent2)
            child.mutate()
            new_population.append(child)

    return new_population[:POPULATION_SIZE]  # Ensure population size is maintained

# Run Evolution and Game
def run_evolution():
    population = [Genome(7, 10, 3) for _ in range(POPULATION_SIZE)]
    best_genome = None
    best_fitness = 0

    for generation in range(GENERATIONS):
        for genome in population:
            genome.fitness = evaluate_fitness(genome)

        population = evolve_population(population)
        generation_best_fitness = max(genome.fitness for genome in population)
        print(f"Generation {generation + 1}, Best Fitness: {generation_best_fitness}")

        # tracking  the best genome/neural network
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_genome = population[np.argmax([genome.fitness for genome in population])]

    return best_genome 


def play_with_best_genome(best_genome):
    game = SnakeGame()
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
    pygame.display.set_caption('Snake Game - Best Genome')
    
    running = True
    while running:
        state = game.get_state()
        action = np.argmax(best_genome.forward(state))
        done, score = game.play_step(action)

        screen.fill((0, 0, 0))
        for segment in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), (segment.x * CELL_SIZE, segment.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, (255, 0, 0), (game.food.x * CELL_SIZE, game.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
        pygame.time.delay(100)  # kontrol game speed

        if done:
            print("Game Over! Final Score:", score)
            running = False
            game.reset()

if __name__ == "__main__":
    #best_genome = run_evolution()
    best_genome = load_genome("best2.obj")
    #save_genome(best_genome, "best.obj")
    play_with_best_genome(best_genome)

