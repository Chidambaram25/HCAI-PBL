# HCAI Project Hub
**Human-Centric Artificial Intelligence** — Course Projects
 
> Chidambaram Muthukumar — 641326
 
A single Django application hosting four interactive projects built as part of the Human-Centric Artificial Intelligence course. Each project explores a different aspect of human-AI interaction, from supervised learning interfaces to preference elicitation.
 
---
 
## Projects
 
### Project 1 — Supervised Learning Interface
An end-to-end interface for supervised learning on user-uploaded datasets.
 
- Upload any CSV dataset and preview the data
- Generate scatter plots with selectable axes and color-coded classes
- Train KNN, Decision Tree, or SVM classifiers
- Control train/test split and hyperparameter values
- View train/test accuracy and a learning curve across hyperparameter values
### Project 2 — Explainability Interface
An interactive explainability dashboard built on the Palmer Penguins dataset.
 
- Visualize decision trees with accuracy and leaf count
- Lambda (λ) slider that selects the model minimizing `acctest + λ·Ω(f)`
- Supports both Decision Tree (Ω = number of leaves) and Logistic Regression (Ω = L2 norm of weights)
- Counterfactual explanation generator — select an example and target class to see what would need to change
- PDP and ALE feature effect plots (written from scratch, no library) with 3 curves per species
### Project 3 — Active Learning for Learning-to-Defer
A human-AI collaborative system for AG News topic classification.
 
- Baseline classifier: TF-IDF + Logistic Regression — **91.53% test accuracy**
- Simulated expert: strong on World/Sports (95%), weak on Business/Sci-Tech (60%)
- Learning-to-defer: confidence-based deferral reaching **92.59%** at optimal threshold τ = 0.6
- Active learning: uncertainty sampling discovers a near-optimal deferral threshold using only 200 expert queries, outperforming random sampling
### Project 4 — Preference Elicitation User Study
A production-ready user study interface comparing two movie preference elicitation methods.
 
- Dataset: IMDB 5000 Movies with 160 features (genres, country, language, content rating, duration, year)
- Preference model: Bradley–Terry extended to rankings via the Plackett–Luce model
- Design 1: 10 rounds of pairwise movie comparisons
- Design 2: 2 rounds of ranking 10 movies via drag-and-drop
- Full user study flow: consent form → instructions → elicitation → recommendations → questionnaire → debrief
- Between-subjects design with random assignment
---
 
## Setup
 
### Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies
### Installation
 
```bash
# Clone the repository
git clone https://github.com/Chidambaram25/HCAI-PBL
cd HCAI-PBL
 
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # Mac/Linux
 
# Install dependencies
pip install -r requirements.txt
 
# Run database migrations
python manage.py migrate
 
# Start the development server
python manage.py runserver
```
 
Then open `http://127.0.0.1:8000/home/` in your browser.
 
### Project 3 — additional setup
 
The AG News dataset is downloaded automatically via Hugging Face on first run.
Trained models are cached in `project3/saved_models/` after the first training run.
 
### Project 4 — additional setup
 
1. Download the [IMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle
2. Place `movie_metadata.csv` in the `data/` folder
3. Run the preprocessing script:
```bash
python feature_extraction.py
```
