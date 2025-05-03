
🧊 Simulated Annealing for Clustering Customers
📖 Overview
This project implements Simulated Annealing (SA) for solving a clustering problem on mall customer data. Instead of using standard k-means, we leverage SA to iteratively improve cluster centroids with probabilistic acceptance of worse solutions, allowing escape from local minima.

Simulated Annealing is inspired by the process of slowly cooling a material to reduce defects, allowing the system to reach a low-energy (optimal) state. In optimization, temperature controls the probability of accepting worse solutions—higher temperatures allow more exploration; as it cools, the search becomes more exploitative.

We experiment with different cooling schedules to see their impact on clustering performance.

🏗️ What does the code do?
✅ Loads preprocessed mall customer data
✅ Initializes cluster centroids using k-means++
✅ Iteratively perturbs centroids and probabilistically accepts them based on SA
✅ Supports 4 cooling schedules:

Exponential cooling

Linear cooling

Very Fast Simulated Reannealing (VCM)

Adaptive cooling (based on recent acceptance rates)

We run SA multiple times for each cooling schedule to analyze:

Convergence speed (how fast cost decreases)

Probability of getting stuck in a poor solution

Average final clustering cost

⚙️ Configurations Compared
We experimented with four configurations, each using a different cooling strategy:


Cooling Type	Cooling Rate
EXPONENTIAL	0.93
LINEAR	0.95
VCM	0.5
ADAPTIVE	0.95
Other parameters (number of clusters, initial temperature, etc.) were held constant.

We ran 10 independent runs for each configuration to compute averages and variability.

📊 Interpretation of Results
1️⃣ Convergence Plot
This shows how quickly each cooling schedule reduces the clustering cost:

Exponential cooling typically showed smooth and fast convergence.

Linear cooling was slower, as temperature dropped more gradually.

VCM cooling led to early aggressive drops but plateaued, showing mixed exploration.

Adaptive cooling had fluctuations depending on acceptance rate adjustment.

👉 Exponential cooling achieved a strong balance between speed and stability.

2️⃣ Probability of Getting Stuck
We measured the proportion of runs that ended with a cost more than 10% higher than the best run for that config:

Exponential cooling: lower probability of getting stuck

VCM and Adaptive: higher likelihood of ending in local minima

Linear cooling: intermediate behavior

👉 Exponential was more reliable across runs; Adaptive and VCM were more erratic.

3️⃣ Average Final Cost
This compares how good the final clusterings were (lower is better):

Exponential cooling achieved the lowest average final cost.

Linear slightly worse.

VCM and Adaptive had higher variability and worse average costs.

📝 Key Takeaways
✅ Exponential cooling consistently provided a good tradeoff between exploration and convergence speed.

✅ VCM and Adaptive cooling did not outperform exponential or linear in this context, suggesting that for this dataset, simpler schedules were sufficient.

✅ Cooling rate tuning is crucial—a too-fast or too-slow temperature drop can hinder performance.

🚀 Running the Code
Place your preprocessed data file as mall_customers_preprocessed.csv.

Run the script:

bash
Copy
Edit
python your_script.py
Visualizations will automatically appear, and statistics will print to the console.

📚 Dependencies
numpy

pandas

matplotlib

tqdm

Install them with:

bash
Copy
Edit
pip install numpy pandas matplotlib tqdm
