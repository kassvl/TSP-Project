# Traveling Salesman Problem (TSP) Solver

This project implements a genetic algorithm solution for the Traveling Salesman Problem (TSP), specifically optimized for the Berlin52 dataset.

## Implementation Details

### Algorithm Components

1. **Initial Population Generation**
   - Greedy solutions from multiple starting points
   - Nearest neighbor solutions
   - Random solutions for diversity
   - Population size: 150

2. **Genetic Operators**
   - Crossover rate: 0.85
   - Mutation rate: 0.1
   - Tournament selection (size: 5)
   - Elitism: 2 best solutions preserved

3. **Local Search Optimization**
   - 2-opt local search
   - Periodic improvement of best solutions
   - Final intensive improvement phase

### Parameters
- Population size: 150
- Number of epochs: 150
- Early stopping: 30 epochs without improvement
- Multiple attempts to find best solution

## Results

Best results achieved:
1. **Best Run**: 7576.25 (0.45% gap to optimal)
2. **Good Run**: 7677.66 (1.80% gap to optimal)
3. **Average Run**: ~7900 (4.74% gap to optimal)

Optimal solution for Berlin52: 7542

## Files Structure

- `tsp_parser.py`: Core implementation of genetic algorithm and TSP operations
- `tsp_visualization.py`: Visualization tools for solutions and convergence
- `tsp_analysis.py`: Analysis utilities and performance metrics
- `berlin52.tsp`: Dataset file containing 52 cities' coordinates

## Visualization Features

1. **Route Visualization**
   - Cities plotted as red dots with labels
   - Route shown as blue lines
   - Clear display of path connections

2. **Convergence History**
   - Plot showing fitness improvement over generations
   - Comparison with optimal solution
   - Early stopping points marked

3. **Animation**
   - Dynamic visualization of route construction
   - Step-by-step path building
   - Clear representation of city connections

## Usage

Run the main visualization script:
```bash
python3 tsp_visualization.py
```

This will:
1. Run multiple attempts of the genetic algorithm
2. Display the best solution found
3. Show convergence history
4. Create an animation of the solution

## Performance

The algorithm consistently finds solutions within 5% of the optimal, with best runs achieving less than 1% gap to optimal. The implementation balances between:
- Solution quality
- Computation speed
- Consistency of results

## Future Improvements

Potential areas for enhancement:
1. Parameter auto-tuning
2. Additional local search strategies
3. Parallel processing for multiple attempts
4. More sophisticated crossover operators
