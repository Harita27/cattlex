<h1 align="center">🐄 CATTLEX – AI POWERED LIVESTOCK HEALTH MONITORING</h1>

<h2>📌 Overview</h2>
<p>
Cattlex is an AI-driven livestock health monitoring system that uses machine learning to detect
abnormalities in cattle based on vital signs and behavioral features.  
It helps farmers, veterinarians, and livestock managers identify potential diseases early.
</p>

<h2>🚀 Key Features</h2>
<ul>
  <li><b>Machine Learning Models:</b> Decision Tree, Random Forest, Naive Bayes, KNN</li>
  <li><b>Real-time Health Prediction</b> based on input symptoms</li>
  <li><b>Model Training & Evaluation</b> with clear performance metrics</li>
  <li><b>Best Model Auto-Saved</b> as <code>model.pkl</code></li>
  <li><b>Confusion Matrix & Reports</b> included for deep analysis</li>
</ul>

<h2>📊 Dataset Details</h2>
<p>The project uses two datasets: <code>Training.csv</code> and <code>Testing.csv</code></p>

<h3>Features Include:</h3>
<ul>
  <li>Temperature</li>
  <li>Respiratory Rate</li>
  <li>Heart Rate</li>
  <li>Feed Intake</li>
  <li>Water Intake</li>
  <li><b>Disease Prognosis (Target)</b></li>
</ul>

<h2>🧠 Technologies Used</h2>
<ul>
  <li><b>Python</b></li>
  <li><b>Scikit-Learn</b> for ML models</li>
  <li><b>Pandas & NumPy</b> for preprocessing</li>
  <li><b>Joblib</b> for saving trained model</li>
</ul>

<h2>🛠 Installation</h2>


pip install pandas numpy scikit-learn joblib
<h2>▶️ How to Run the Project</h2>
python livestock_health.py
<h2>🔍 Example Prediction Code</h2>
import joblib
import numpy as np

model = joblib.load("model.pkl")

sample = np.array([[39.5, 110, 32, 0.4, 12, 30]])
print(model.predict(sample))
<h2>🏗 Project Structure</h2>
├── livestock_health.py<br>
├── confusion matrix.py<br>
├── Cattle_Disease_Project_ML.ipynb<br>
├── livestock_health_analysis.ipynb<br>
├── model.pkl<br>
├── database.db<br>
└── readme.md<br>

<h2>🌟 Why This Project Stands Out</h2> <ul> <li>Uses multiple ML models and compares them scientifically</li> <li>Auto-selects the best performing algorithm</li> <li>Designed for <b>real-world livestock health monitoring</b></li> <li>Excellent showcase project for <b>Machine Learning skills</b></li> </ul> <h2>📬 Contact</h2> <p>If you’d like to know more or collaborate, feel free to reach out!</p>
