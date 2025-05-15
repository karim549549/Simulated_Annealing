import pandas as pd
import numpy as np
import random
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkFont

df = pd.read_csv('mall_customers_preprocessed.csv')
print(df.head(5))

class CoolingType(Enum):
    EXPONENTIAL = auto()
    LINEAR = auto()
    VCM = auto()
    ADAPTIVE = auto()

@dataclass
class SAConfig:
    num_clusters: int = 3
    initial_temp: float = 1000.0
    cooling_rate: float = 0.95
    cooling_type: CoolingType = CoolingType.EXPONENTIAL
    max_iter: int = 1000
    termination_cost: float = 0.0
    local_search_freq: int = 0
    perturbation_scale: float = 0.5
    min_temp: float = 1.0  # Minimum temperature threshold


class SimulatedAnnealingClustering:
    def __init__(self, objective, data: np.ndarray, config: SAConfig):
        self.objective = objective
        self.data = data
        self.config = config
        self.temp = config.initial_temp
        self.iteration = 0
        self.acceptance_history = []

        # Initialize centroids using k-means++ style initialization
        n_samples, n_features = data.shape
        self.curr_solution = self._initialize_centroids(n_samples, n_features)
        self.curr_cost = objective(self.curr_solution)
        self.best_solution = self.curr_solution.copy()
        self.best_cost = self.curr_cost
        self.cost_history = [self.curr_cost]

    def _initialize_centroids(self, n_samples: int, n_features: int) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm"""
        centroids = np.zeros((self.config.num_clusters, n_features))

        # First centroid: random data point
        centroids[0] = self.data[np.random.choice(n_samples)]

        for i in range(1, self.config.num_clusters):
            # Calculate distances to nearest centroid
            distances = np.min(np.sum((self.data[:, np.newaxis] - centroids[:i]) ** 2, axis=2), axis=1)
            # Select next centroid with probability proportional to distance squared
            probabilities = distances / np.sum(distances)
            centroids[i] = self.data[np.random.choice(n_samples, p=probabilities)]

        return centroids

    def _perturb_centroids(self) -> np.ndarray:
        """Generate new solution with temperature-scaled noise"""
        noise = np.random.normal(
            scale=self.config.perturbation_scale * (self.temp / self.config.initial_temp),
            size=self.curr_solution.shape
        )
        # Clip to ensure we stay within data bounds
        return np.clip(
            self.curr_solution + noise,
            self.data.min(axis=0),
            self.data.max(axis=0)
        )

    def _local_search(self) -> np.ndarray:
        """K-means style centroid refinement"""
        distances = np.linalg.norm(self.data[:, np.newaxis] - self.curr_solution, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = []
        for i in range(self.config.num_clusters):
            cluster_points = self.data[labels == i]
            if len(cluster_points) == 0:
                # Empty cluster: randomly pick a new centroid from data
                new_centroids.append(self.data[np.random.choice(len(self.data))])
            else:
                new_centroids.append(cluster_points.mean(axis=0))

        return np.array(new_centroids)

    def _accept_solution(self, new_cost: float) -> bool:
        """Metropolis acceptance criterion"""
        if new_cost < self.curr_cost:
            return True

        delta = new_cost - self.curr_cost
        acceptance_prob = np.exp(-delta / self.temp)
        self.acceptance_history.append(acceptance_prob)
        return random.random() < acceptance_prob

    def _update_temperature(self, iteration: int):
        """Update temperature according to cooling schedule"""
        if self.config.cooling_type == CoolingType.EXPONENTIAL:
            self.temp = self.config.initial_temp * (self.config.cooling_rate ** iteration)
        elif self.config.cooling_type == CoolingType.LINEAR:
            self.temp = self.config.initial_temp * (1 - iteration / self.config.max_iter)
        elif self.config.cooling_type == CoolingType.VCM:
            # Very Fast Simulated Reannealing schedule
            d = self.data.shape[1]  # dimensionality
            self.temp = self.config.initial_temp * np.exp(
                -self.config.cooling_rate * (iteration ** (-1 / d))
            )
        elif self.config.cooling_type == CoolingType.ADAPTIVE:
            # Adaptive cooling based on acceptance rate
            window_size = min(50, iteration + 1)
            recent_acceptance = np.mean(self.acceptance_history[-window_size:]) if self.acceptance_history else 1.0
            if recent_acceptance < 0.2:  # Too few acceptances
                self.temp *= 1.05  # Slightly increase temperature
            elif recent_acceptance > 0.5:  # Too many acceptances
                self.temp *= 0.95  # Slightly decrease temperature
            else:
                self.temp *= self.config.cooling_rate

        # Ensure temperature doesn't go below minimum
        self.temp = max(self.temp, self.config.min_temp)

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run the simulated annealing clustering algorithm

        Returns:
            Tuple of (best_solution, best_cost)
        """
        for self.iteration in range(1, self.config.max_iter):
            # 1. Generate new candidate solution
            new_solution = self._perturb_centroids()

            # 2. Optional local search refinement
            if self.config.local_search_freq > 0 and self.iteration % self.config.local_search_freq == 0:
                new_solution = self._local_search()

            # 3. Evaluate new solution
            new_cost = self.objective(new_solution)

            # 4. Metropolis acceptance criterion
            if self._accept_solution(new_cost):
                self.curr_solution = new_solution
                self.curr_cost = new_cost

                # Update best solution if improved
                if new_cost < self.best_cost:
                    self.best_solution = new_solution.copy()
                    self.best_cost = new_cost

            # 5. Record history for analysis
            self.cost_history.append(self.curr_cost)

            # 6. Update temperature
            self._update_temperature(self.iteration)

            # 7. Early termination if cost threshold reached
            if self.curr_cost <= self.config.termination_cost:
                break

        return self.best_solution, self.best_cost


# Example usage
def kmeans_cost(centroids: np.ndarray, data: np.ndarray) -> float:
    """Calculate sum of squared distances to nearest centroid"""
    distances = np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2)
    return np.sum(np.min(distances, axis=1))


# Prepare data
data = df.values

# Create configuration
config = SAConfig(
    num_clusters=5,
    cooling_type=CoolingType.EXPONENTIAL,
    cooling_rate=0.93,
    max_iter=500
)

# Run clustering
sa = SimulatedAnnealingClustering(
    objective=lambda x: kmeans_cost(x, data),
    data=data,
    config=config
)
centroids, cost = sa.run()

print(f"Best cost: {cost:.2f}")
print("Centroids:\n", centroids)




def run_experiments(data: np.ndarray, configs: List[SAConfig],
                    num_runs: int = 1000000, objective=None) -> Dict:
    """
    Run multiple SA configurations and collect statistics

    Args:
        data: Input data matrix
        configs: List of SAConfig objects to test
        num_runs: Number of runs per configuration
        objective: Cost function (default: kmeans_cost)

    Returns:
        Dictionary containing all results and statistics
    """
    if objective is None:
        objective = lambda x: kmeans_cost(x, data)

    results = {
        'configs': configs,
        'all_runs': [],
        'statistics': []
    }

    # Run each configuration multiple times
    for config in tqdm(configs, desc="Testing configurations"):
        config_runs = []
        for run in range(num_runs):
            sa = SimulatedAnnealingClustering(objective, data, config)
            centroids, final_cost = sa.run()

            config_runs.append({
                'cost_history': sa.cost_history,
                'final_cost': final_cost,
                'centroids': centroids,
                'acceptance_history': sa.acceptance_history
            })

        # Calculate statistics for this configuration
        final_costs = [run['final_cost'] for run in config_runs]
        avg_cost = np.mean(final_costs)
        std_cost = np.std(final_costs)

        results['all_runs'].append(config_runs)
        results['statistics'].append({
            'config': config,
            'avg_cost': avg_cost,
            'std_cost': std_cost,
            'min_cost': np.min(final_costs),
            'max_cost': np.max(final_costs),
            'stuck_probability': np.mean(np.array(final_costs) > 1.1 * np.min(final_costs))
        })

    return results


def plot_convergence_comparison(results: Dict):
    """Plot average convergence for each configuration"""
    plt.figure(figsize=(12, 6))

    for i, config_runs in enumerate(results['all_runs']):
        # Get all cost histories for this config
        histories = [run['cost_history'] for run in config_runs]

        # Pad histories to equal length for averaging
        max_len = max(len(h) for h in histories)
        padded = [h + [h[-1]] * (max_len - len(h)) for h in histories]

        # Calculate mean and std across runs
        mean_history = np.mean(padded, axis=0)
        std_history = np.std(padded, axis=0)

        # Plot with confidence bands
        config_name = results['configs'][i].cooling_type.name
        plt.plot(mean_history, label=f"{config_name}")
        plt.fill_between(
            range(len(mean_history)),
            mean_history - std_history,
            mean_history + std_history,
            alpha=0.2
        )

    plt.title("Average Convergence Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_stuck_probability(results: Dict):
    """Visualize probability of getting stuck in local minima"""
    stats = results['statistics']
    config_names = [s['config'].cooling_type.name for s in stats]
    stuck_probs = [s['stuck_probability'] for s in stats]
    avg_final_costs = [s['avg_cost'] for s in stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Probability of getting stuck
    ax1.bar(config_names, stuck_probs)
    ax1.set_title("Probability of Getting Stuck in Local Minima")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)

    # Average final costs
    ax2.bar(config_names, avg_final_costs)
    ax2.set_title("Average Final Cost Across Runs")
    ax2.set_ylabel("Cost")

    plt.tight_layout()
    plt.show()

def start_experiment():
    messagebox.showinfo("Experiment", "Running experiments... This may take a while!")

    # Run experiments (same as your __main__ block)
    results = run_experiments(data, configs, num_runs=10)

    # Show plots
    plot_convergence_comparison(results)
    plot_stuck_probability(results)

    # Print summary in a popup
    summary = ""
    for stat in results['statistics']:
        summary += f"\n{stat['config'].cooling_type.name}:\n"
        summary += f"  Avg Cost: {stat['avg_cost']:.2f} ± {stat['std_cost']:.2f}\n"
        summary += f"  Stuck Probability: {stat['stuck_probability']:.2f}\n"
        summary += f"  Best Run: {stat['min_cost']:.2f}\n"
        summary += f"  Worst Run: {stat['max_cost']:.2f}\n"

    messagebox.showinfo("Results Summary", summary)

# Example usage
if __name__ == "__main__":
    # Define configurations to test
    configs = [
        SAConfig(cooling_type=CoolingType.EXPONENTIAL, cooling_rate=0.93),
        SAConfig(cooling_type=CoolingType.LINEAR, cooling_rate=0.95),
        SAConfig(cooling_type=CoolingType.VCM, cooling_rate=0.5),
        SAConfig(cooling_type=CoolingType.ADAPTIVE, cooling_rate=0.95)
    ]

    # GUI window
    root = tk.Tk()
    root.title("Simulated Annealing Clustering")
    root.geometry("500x300")
    root.configure(bg="#1e1e1e")  # dark background

    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 3
    root.geometry(f"+{x}+{y}")

    # Fonts
    header_font = tkFont.Font(family="Helvetica", size=20, weight="bold")
    button_font = tkFont.Font(family="Helvetica", size=14)

    # Header label
    header_label = tk.Label(root,
                            text="Simulated Annealing Clustering",
                            bg="#1e1e1e",
                            fg="#f0f0f0",  # light text
                            font=header_font)
    header_label.pack(pady=40)

    # Start button
    start_button = tk.Button(root,
                             text="Run Clustering Experiment",
                             command=start_experiment,
                             font=button_font,
                             bg="#333333",  # dark button background
                             fg="#ffffff",  # button text color
                             activebackground="#444444",  # button hover
                             activeforeground="#ffffff",
                             relief="raised",
                             bd=3,
                             height=2,
                             width=25)
    start_button.pack(pady=20)

    root.mainloop()

