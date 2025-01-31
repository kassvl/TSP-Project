import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time

@dataclass
class City:
    id: int
    x: float
    y: float

@dataclass
class TSPData:
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.city_names = [
            "Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt",
            "Stuttgart", "Düsseldorf", "Leipzig", "Dortmund", "Essen",
            "Bremen", "Dresden", "Hanover", "Nuremberg", "Duisburg",
            "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Münster",
            "Karlsruhe", "Mannheim", "Augsburg", "Wiesbaden", "Gelsenkirchen",
            "Mönchengladbach", "Braunschweig", "Kiel", "Chemnitz", "Aachen",
            "Halle", "Magdeburg", "Freiburg", "Krefeld", "Lübeck",
            "Oberhausen", "Erfurt", "Mainz", "Rostock", "Kassel",
            "Hagen", "Hamm", "Saarbrücken", "Mülheim", "Potsdam",
            "Ludwigshafen", "Oldenburg", "Leverkusen", "Osnabrück", "Solingen",
            "Heidelberg", "Darmstadt"
        ]

def parse_tsp_file(filename: str) -> TSPData:
    cities = []
    dimension = 0
    coordinates_section = False
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  
                continue
            if line.startswith("DIMENSION"):
                dimension = int(line.split()[-1])
            elif line.startswith("NODE_COORD_SECTION"):
                coordinates_section = True
            elif coordinates_section and line != "EOF":
                try:
                    id_, x, y = map(float, line.split())
                    cities.append(City(int(id_), x, y))
                except ValueError:
                    continue  
    
    return TSPData(cities)

def calculate_distance(city1: City, city2: City) -> float:
    return np.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)

def calculate_fitness(tsp_data: TSPData, solution: np.ndarray) -> float:
    total_distance = 0
    for i in range(len(solution)):
        city1 = tsp_data.cities[solution[i] - 1]
        city2 = tsp_data.cities[solution[(i + 1) % len(solution)] - 1]
        total_distance += calculate_distance(city1, city2)
    return total_distance

def generate_random_solution(tsp_data: TSPData) -> np.ndarray:
    solution = list(range(1, tsp_data.cities[-1].id + 1))
    random.shuffle(solution)
    return np.array(solution)

def generate_nearest_neighbor_solution(tsp_data: TSPData, start_city: int) -> np.ndarray:
    unvisited = set(range(1, tsp_data.cities[-1].id + 1))
    current_city = start_city + 1
    solution = [current_city]
    unvisited.remove(current_city)
    
    while unvisited:
        current = tsp_data.cities[current_city - 1]
        next_city = min(unvisited,
                       key=lambda x: calculate_distance(current, tsp_data.cities[x - 1]))
        solution.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    return np.array(solution)

def greedy_algorithm(tsp_data: TSPData, start_city: int) -> np.ndarray:
    solution = []
    unvisited = set(range(1, tsp_data.cities[-1].id + 1))
    current_city = start_city + 1
    solution.append(current_city)
    unvisited.remove(current_city)
    
    while unvisited:
        current = tsp_data.cities[current_city - 1]
        next_city = min(unvisited,
                       key=lambda x: calculate_distance(current, tsp_data.cities[x - 1]))
        solution.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    return np.array(solution)

def ordered_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = np.zeros(size, dtype=int)
    child[start:end] = parent1[start:end]
    
    remaining = [x for x in parent2 if x not in child[start:end]]
    child[:start] = remaining[:start]
    child[end:] = remaining[start:]
    
    return child

def swap_mutation(solution: np.ndarray, mutation_rate: float) -> np.ndarray:
    mutated = solution.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def inversion_mutation(solution: np.ndarray, mutation_rate: float) -> np.ndarray:
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(solution)), 2))
        solution[start:end] = solution[start:end][::-1]
    return solution

def tournament_selection(population, tournament_size: int) -> np.ndarray:
    tournament = random.sample(population.solutions, tournament_size)
    return min(tournament, key=lambda x: calculate_fitness(population.tsp_data, x))

class Population:
    def __init__(self, solutions: List[np.ndarray], tsp_data: TSPData):
        self.solutions = solutions
        self.tsp_data = tsp_data
        self.best_solution = None
        self.best_fitness = float('inf')
        self.update_best()
    
    def update_best(self):
        for solution in self.solutions:
            fitness = calculate_fitness(self.tsp_data, solution)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()

class PlotManager:
    def __init__(self):
        plt.ion()  
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Genetic Algorithm Progress', fontsize=16)
        
        self.generations = []
        self.best_distances = []
        self.optimal_distance = 7542  
        
        self.ax1.set_title('Best Distance per Generation')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Distance')
        self.ax1.grid(True)
        
        self.ax2.set_title('Gap to Optimal Solution (%)')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Gap (%)')
        self.ax2.grid(True)
        
        self.best_line, = self.ax1.plot([], [], 'g-', label='Best Distance')
        self.optimal_line, = self.ax1.plot([], [], 'r--', label=f'Optimal ({self.optimal_distance})')
        self.gap_line, = self.ax2.plot([], [], 'r-', label='Gap to Optimal')
        
        self.ax1.legend()
        self.ax2.legend()
        
        plt.tight_layout()
        self.fig.subplots_adjust(top=0.9)
    
    def update(self, generation: int, best_distance: float, population: Population):
        self.generations.append(generation)
        self.best_distances.append(best_distance)
        
        gap = ((best_distance - self.optimal_distance) / self.optimal_distance) * 100
        
        self.best_line.set_data(self.generations, self.best_distances)
        self.optimal_line.set_data([min(self.generations), max(self.generations)], 
                                 [self.optimal_distance, self.optimal_distance])
        self.gap_line.set_data(self.generations, 
                              [(d - self.optimal_distance) / self.optimal_distance * 100 
                               for d in self.best_distances])
        
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        self.fig.suptitle(f'Genetic Algorithm Progress\nBest: {best_distance:.2f}, '
                         f'Gap: {gap:.2f}%, Generation: {generation}')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)

class GeneticAlgorithm:
    def __init__(self, tsp_data: TSPData, pop_size: int, crossover_rate: float,
                 mutation_rate: float, tournament_size: int, elitism: int):
        self.tsp_data = tsp_data
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.initial_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.best_solution = None
        self.best_fitness = float('inf')
        self.generations_without_improvement = 0
    
    def create_epoch(self, current_population: Population) -> Population:
        new_solutions = []
        
        if self.generations_without_improvement > 10:  
            self.mutation_rate = min(0.3, self.mutation_rate * 1.05)  
        elif self.generations_without_improvement > 20:  
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)  
        else:
            self.mutation_rate = max(self.initial_mutation_rate, 
                                   self.mutation_rate * 0.95)  
        
        if self.elitism > 0:
            elite_solutions = sorted(current_population.solutions, 
                                   key=lambda x: calculate_fitness(current_population.tsp_data, x))[:self.elitism]
            for solution in elite_solutions:
                new_solutions.append(solution.copy())
        
        while len(new_solutions) < self.pop_size:
            parent1 = tournament_selection(current_population, self.tournament_size)
            parent2 = tournament_selection(current_population, self.tournament_size)
            
            if np.random.random() < self.crossover_rate:
                offspring = ordered_crossover(parent1, parent2)
            else:
                fitness1 = calculate_fitness(current_population.tsp_data, parent1)
                fitness2 = calculate_fitness(current_population.tsp_data, parent2)
                offspring = parent1.copy() if fitness1 < fitness2 else parent2.copy()
            
            if np.random.random() < self.mutation_rate:
                r = np.random.random()
                if r < 0.4:  
                    offspring = inversion_mutation(offspring, self.mutation_rate)
                elif r < 0.7:  
                    offspring = swap_mutation(offspring, self.mutation_rate)
                else:  
                    offspring = inversion_mutation(offspring, self.mutation_rate)
                    offspring = swap_mutation(offspring, self.mutation_rate * 0.5)
            
            new_solutions.append(offspring)
        
        new_population = Population(new_solutions, current_population.tsp_data)
        
        if new_population.best_fitness < self.best_fitness:
            self.best_fitness = new_population.best_fitness
            self.best_solution = new_population.best_solution.copy()
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        
        return new_population

def generate_initial_population(tsp_data: TSPData, pop_size: int, include_greedy: bool = True) -> Population:
    solutions = []
    
    if include_greedy:
        for start in range(len(tsp_data.cities)):
            solution = greedy_algorithm(tsp_data, start)
            solutions.append(solution)
            
            if len(solutions) <= 15:  
                for _ in range(3):  
                    mutated = solution.copy()
                    mutated = inversion_mutation(mutated, 0.3)  
                    mutated = swap_mutation(mutated, 0.2)  
                    solutions.append(mutated)
    
    num_nn = min(pop_size // 3, len(tsp_data.cities))  
    for _ in range(num_nn):
        start = np.random.randint(len(tsp_data.cities))
        solution = generate_nearest_neighbor_solution(tsp_data, start)
        solutions.append(solution)
        
        if np.random.random() < 0.5:  
            mutated = solution.copy()
            mutated = inversion_mutation(mutated, 0.3)
            solutions.append(mutated)
    
    solutions.sort(key=lambda x: calculate_fitness(tsp_data, x))
    best_solutions = solutions[:pop_size//2]
    
    while len(solutions) < pop_size:
        if np.random.random() < 0.7 and best_solutions:  
            base = best_solutions[np.random.randint(len(best_solutions))].copy()
            if np.random.random() < 0.5:
                solution = inversion_mutation(base, 0.3)
            else:
                solution = swap_mutation(base, 0.2)
        else:
            solution = generate_random_solution(tsp_data)
        solutions.append(solution)
    
    solutions.sort(key=lambda x: calculate_fitness(tsp_data, x))
    return Population(solutions[:pop_size], tsp_data)

def run_genetic_algorithm(tsp_data: TSPData, num_epochs: int, pop_size: int,
                         crossover_rate: float, mutation_rate: float,
                         tournament_size: int, elitism: int,
                         early_stopping: Optional[int] = None):
    
    ga = GeneticAlgorithm(tsp_data, pop_size, crossover_rate, mutation_rate,
                         tournament_size, elitism)
    
    population = generate_initial_population(tsp_data, pop_size, include_greedy=True)
    history = []
    
    plot_manager = PlotManager()
    
    print(f"Initial Best Distance: {population.best_fitness:.2f}")
    
    try:
        no_improvement_count = 0
        best_fitness_ever = float('inf')
        
        for epoch in range(num_epochs):
            if no_improvement_count > 20:  
                ga.mutation_rate = min(0.5, ga.mutation_rate * 1.2)  
            else:
                ga.mutation_rate = max(0.1, ga.mutation_rate * 0.95)  
            
            population = ga.create_epoch(population)
            history.append(population.best_fitness)
            
            plot_manager.update(epoch, population.best_fitness, population)
            
            if population.best_fitness < best_fitness_ever:
                best_fitness_ever = population.best_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Best Distance = {population.best_fitness:.2f}")
            
            if early_stopping and no_improvement_count >= early_stopping:
                print(f"\nStopping early - No improvement for {early_stopping} epochs")
                break
    
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    finally:
        plot_manager.close()
    
    return ga.best_solution, ga.best_fitness, history

def test_berlin52():
    print("\nTesting on berlin52.tsp")
    print("=" * 50 + "\n")
    
    tsp_data = parse_tsp_file("berlin52.tsp")
    
    print("Starting Genetic Algorithm:")
    print(f"Population Size: 300")  
    print(f"Number of Epochs: 500")  
    
    solution, fitness, history = run_genetic_algorithm(
        tsp_data,
        num_epochs=500,  
        pop_size=300,    
        crossover_rate=0.95,
        mutation_rate=0.2,  
        tournament_size=5,
        elitism=15,  
        early_stopping=100
    )
    
    print(f"\nFinal Best Distance: {fitness:.2f}")
    print("\nBest Result Found:")
    print(f"Distance: {fitness:.2f}")
    print("Tour: " + " -> ".join(tsp_data.city_names[i-1] for i in solution))
    
    gap = ((fitness - 7542) / 7542) * 100
    print(f"Gap to Optimal (7542): {gap:.2f}%")
    
    print("\nAnimated visualization of the solution:")
    visualizer = TSPVisualizer(tsp_data)
    visualizer.animate_solution(solution, f"German Cities TSP Solution (Total Distance: {fitness:.2f})")

class TSPVisualizer:
    def __init__(self, tsp_data: TSPData):
        self.tsp_data = tsp_data
        self.x_coords = [city.x for city in tsp_data.cities]
        self.y_coords = [city.y for city in tsp_data.cities]
        plt.style.use('default')
    
    def calculate_partial_distance(self, solution: List[int], current_pos: int) -> float:
        distance = 0
        for i in range(current_pos):
            city1 = self.tsp_data.cities[solution[i] - 1]
            city2 = self.tsp_data.cities[solution[i + 1] - 1]
            distance += calculate_distance(city1, city2)
        return distance

    def plot_single_solution(self, solution: List[int], title: str = "TSP Solution", 
                           figsize: Tuple[int, int] = (12, 8)):
        plt.figure(figsize=figsize)
        self._plot_solution(solution)
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    def _plot_solution(self, solution: List[int], current_pos: int = None):
        plt.scatter(self.x_coords, self.y_coords, c='lightcoral', s=100, zorder=2, alpha=0.5)
        
        for i, (x, y) in enumerate(zip(self.x_coords, self.y_coords)):
            plt.annotate(self.tsp_data.city_names[i], (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        if current_pos is not None:
            for i in range(current_pos):
                start = solution[i] - 1
                end = solution[i + 1] - 1
                
                plt.plot([self.x_coords[start], self.x_coords[end]], 
                        [self.y_coords[start], self.y_coords[end]], 
                        'b-', alpha=0.6, zorder=1, linewidth=2)
                
                plt.scatter(self.x_coords[start], self.y_coords[start], 
                          c='limegreen', s=120, zorder=3, edgecolor='darkgreen')
                
                plt.annotate(f'{i+1}', (self.x_coords[start], self.y_coords[start]),
                           xytext=(-3, -3), textcoords='offset points',
                           color='white', fontweight='bold', fontsize=8)
            
            if current_pos < len(solution) - 1:
                current_city = solution[current_pos] - 1
                plt.scatter(self.x_coords[current_city], self.y_coords[current_city], 
                          c='yellow', s=150, zorder=4, edgecolor='black', linewidth=2)
                plt.annotate(f'Current: {self.tsp_data.city_names[current_city]}', 
                           (self.x_coords[current_city], self.y_coords[current_city]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
            else:
                last_city = solution[current_pos] - 1
                plt.scatter(self.x_coords[last_city], self.y_coords[last_city], 
                          c='limegreen', s=120, zorder=3, edgecolor='darkgreen')
                plt.annotate(f'{current_pos+1}', (self.x_coords[last_city], self.y_coords[last_city]),
                           xytext=(-3, -3), textcoords='offset points',
                           color='white', fontweight='bold', fontsize=8)
                
                first_city = solution[0] - 1
                plt.plot([self.x_coords[last_city], self.x_coords[first_city]], 
                        [self.y_coords[last_city], self.y_coords[first_city]], 
                        'b-', alpha=0.6, zorder=1, linewidth=2)
        else:
            for i in range(len(solution)):
                start = solution[i] - 1
                end = solution[(i + 1) % len(solution)] - 1
                plt.plot([self.x_coords[start], self.x_coords[end]], 
                        [self.y_coords[start], self.y_coords[end]], 
                        'b-', alpha=0.6, zorder=1)
                plt.scatter(self.x_coords[start], self.y_coords[start], 
                          c='limegreen', s=120, zorder=3, edgecolor='darkgreen')
                plt.annotate(f'{i+1}', (self.x_coords[start], self.y_coords[start]),
                           xytext=(-3, -3), textcoords='offset points',
                           color='white', fontweight='bold', fontsize=8)
    
    def animate_solution(self, solution: List[int], title: str = "TSP Solution Animation"):
        fig = plt.figure(figsize=(12, 8))
        plt.grid(True)
        
        def update(frame):
            plt.clf()  
            plt.grid(True)
            
            plt.title(f"{title}\nStep {frame+1}/{len(solution)}: "
                     f"Visiting City {solution[frame]}", pad=20)
            
            self._plot_solution(solution, frame)
            
            margin = 20
            plt.xlim(min(self.x_coords) - margin, max(self.x_coords) + margin)
            plt.ylim(min(self.y_coords) - margin, max(self.y_coords) + margin)
            
            legend_elements = [
                plt.scatter([], [], c='lightcoral', s=100, label='Unvisited Cities'),
                plt.scatter([], [], c='limegreen', s=100, label='Visited Cities'),
                plt.scatter([], [], c='yellow', s=100, label='Current City'),
                plt.plot([], [], 'b-', label='Path')[0]
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        anim = FuncAnimation(fig, update, frames=len(solution), 
                           interval=800, repeat=False)
        plt.show()

if __name__ == "__main__":
    test_berlin52()
