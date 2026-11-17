"""
Evolutionary & Genetic Algorithm Optimizer for Intrusion Detection Systems (IDS).
Author: Jeremiah Okino (okino007)
Description: Evolves hyper-parameters and feature weights to optimize IDS classification accuracy.
"""

import random
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IDSGeneticOptimizer:
    """
    Implements evolutionary selection, crossover, and mutation operators to fine-tune
    IDS decision thresholds and feature weights for optimal network intrusion detection.
    """
    def __init__(self, population_size: int = 50, generations: int = 25, mutation_rate: float = 0.08):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_individual(self) -> List[float]:
        """Generates a random chromosome representing IDS decision weights and threshold."""
        # [weight_packet_rate, weight_entropy, weight_payload_size, detection_threshold]
        return [round(random.uniform(0.1, 1.0), 4) for _ in range(4)]

    def fitness_function(self, chromosome: List[float]) -> float:
        """
        Simulates evaluation of IDS performance (F1-Score / Accuracy) against validation traffic.
        """
        w_rate, w_entropy, w_payload, threshold = chromosome
        # Simulated heuristic fitness score maximizing detection true-positives while minimizing false-positives
        base_score = 0.85 + (0.05 * w_rate) + (0.04 * w_entropy) - (0.02 * abs(threshold - 0.65))
        return round(min(1.0, max(0.0, base_score)), 4)

    def evolve_population(self) -> Tuple[List[float], float]:
        """Runs the evolutionary loop across generations to find optimal IDS parameters."""
        population = [self.initialize_individual() for _ in range(self.population_size)]
        
        best_overall = None
        best_fitness = -1.0

        for gen in range(self.generations):
            scored_pop = [(ind, self.fitness_function(ind)) for ind in population]
            scored_pop.sort(key=lambda x: x[1], reverse=True)

            current_best, current_fit = scored_pop[0]
            if current_fit > best_fitness:
                best_fitness = current_fit
                best_overall = current_best

            # Selection & Elitism (keep top 20%)
            elite_count = int(self.population_size * 0.2)
            next_generation = [ind for ind, fit in scored_pop[:elite_count]]

            # Breed new individuals via crossover and mutation
            while len(next_generation) < self.population_size:
                parent_a = random.choice(scored_pop[:15])[0]
                parent_b = random.choice(scored_pop[:15])[0]
                
                # Single-point crossover
                crossover_point = random.randint(1, 3)
                child = parent_a[:crossover_point] + parent_b[crossover_point:]

                # Mutation
                if random.random() < self.mutation_rate:
                    mutated_idx = random.randint(0, 3)
                    child[mutated_idx] = round(random.uniform(0.1, 1.0), 4)

                next_generation.append(child)

            population = next_generation

        logging.info(f"Evolution complete. Best Fitness Score: {best_fitness}")
        return best_overall, best_fitness
