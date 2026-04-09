# 🤖 AI-Powered ToDo List Manager — CLI Application

A complete CLI task management system built in Python that applies **6 real AI/ML algorithms** learned in the Fundamentals of AI and ML course — all from scratch using only Python's standard library.

---

## 🧠 AI / ML Algorithms Used

| # | Module | Algorithm | Purpose |
|---|--------|-----------|---------|
| 1 | Priority Predictor | **Naïve Bayes Classifier** | Predicts task priority (high/medium/low) from title words |
| 2 | Category Classifier | **TF-IDF + Cosine Similarity** | Suggests task category based on text similarity |
| 3 | Task Clusterer | **K-Means Clustering** | Groups similar tasks into clusters |
| 4 | Anomaly Detector | **Z-Score Statistical Analysis** | Flags overdue tasks and unusual productivity days |
| 5 | Workload Forecaster | **Linear Regression (OLS)** | Predicts new tasks per day for next 7 days |
| 6 | Smart Recommender | **Urgency Scoring + Frequency Mining** | Ranks tasks by computed urgency score |

> All algorithms are implemented **from scratch** — no scikit-learn or external ML libraries used.

---

## ✅ Features

- 🤖 **AI Priority Prediction** — type a title, Naïve Bayes predicts priority automatically
- 🤖 **AI Category Suggestion** — TF-IDF cosine similarity assigns the best category
- 🤖 **Smart Recommendations** — urgency-scored task suggestions for today
- 🤖 **K-Means Clustering** — visual grouping of tasks by similarity
- 🤖 **Anomaly Detection** — Z-score flags overdue outliers and productivity bursts
- 🤖 **Workload Forecast** — linear regression predicts upcoming task load
- ✅ Full CRUD operations (Add, View, Update, Delete, Complete)
- 🎨 Color-coded priorities, overdue indicators, interactive menus
- 💾 SQLite persistent storage
- 📤📥 JSON export / import

---

## 🛠️ Requirements

- Python **3.7+**
- **No external libraries required** — pure Python standard library only

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ai-todo-manager.git
cd ai-todo-manager
```

### 2. Run the application

```bash
python todo.py
```

A `tasks.db` SQLite database is auto-created on first run.

---

## 🖥️ Usage

### Interactive Mode

```bash
python todo.py
```

Main menu:

```
  ─── Main Menu ───────────────────────────────────────────

  TASK OPERATIONS
  1)             Add Task  (AI predicts priority + category)
  2)             List Pending Tasks
  3)             List All Tasks
  4)             View Task Details
  5)             Mark Task Complete
  6)             Update Task
  7)             Delete Task
  8)             Search Tasks

  AI / ML FEATURES
  9)             🤖 Smart Recommendations     (Urgency Scoring)
  10)            🤖 Analyze New Task Title     (NB + TF-IDF)
  11)            🤖 Task Clustering            (K-Means)
  12)            🤖 Anomaly Detection          (Z-Score)
  13)            🤖 Workload Forecast          (Linear Regression)

  DATA & REPORTS
  14)            Statistics Dashboard
  15)            Export Tasks (JSON)
  16)            Import Tasks (JSON)
  0)             Exit
```

---

### Direct CLI Commands

```bash
python todo.py                        # Interactive menu
python todo.py add "Task title"       # Add task (AI auto-predicts priority + category)
python todo.py list                   # List pending tasks
python todo.py done <id>              # Mark task complete
python todo.py delete <id>            # Delete task
python todo.py recommend              # AI smart recommendations
python todo.py forecast               # Workload forecast (linear regression)
python todo.py cluster                # K-Means task clustering
python todo.py anomaly                # Z-score anomaly report
python todo.py stats                  # Statistics dashboard
python todo.py help                   # Help
```

#### Options for `add`:

```bash
python todo.py add "Study for exam" --priority high --category Study --due 2025-06-15 --desc "Chapters 3-5"
```

| Flag | Values | Default |
|------|--------|---------|
| `--priority` | `high`, `medium`, `low` | **AI predicted** |
| `--category` | Any string | **AI predicted** |
| `--due` | `YYYY-MM-DD` | None |
| `--desc` | Any string | Empty |

> **Leave `--priority` and `--category` blank** to let the AI predict them from your title!

---

## 🧠 How the AI Works

### Naïve Bayes Classifier (Priority Prediction)

Trains on past task titles/descriptions. For a new task, computes:

```
P(priority | words) ∝ P(priority) × Π P(word | priority)
```

Uses Laplace (add-1) smoothing for unseen words. Converts log-probabilities to a softmax probability distribution shown as a bar chart.

### TF-IDF + Cosine Similarity (Category)

Builds a TF-IDF weighted centroid vector for each category from past tasks. New tasks are classified by cosine similarity to these centroids:

```
sim(A, B) = (A·B) / (|A| × |B|)
```

### K-Means Clustering

Feature vector per task: `[priority_score, days_until_due, completed_flag, title_length]`
Iteratively assigns tasks to nearest centroid and updates centroids until convergence.

### Z-Score Anomaly Detection

Computes mean (μ) and standard deviation (σ) of overdue gaps and daily completion counts. Flags values where `|x - μ| / σ > threshold`.

### Linear Regression (OLS)

Fits `y = w*x + b` using closed-form solution:
```
w = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
b = ȳ - w × x̄
```

Predicts expected new tasks per day for the next 7 days.

### Urgency Scoring (Recommender)

```
urgency_score = priority_weight / (days_remaining + 1)
```

Higher scores = tasks that need attention soonest.

---

## 📁 Project Structure

```
ai-todo-manager/
│
├── todo.py              # Complete application — all AI/ML + CLI logic
├── tasks.db             # SQLite database (auto-created)
├── sample_tasks.json    # Sample data for import feature
└── README.md            # This file
```

---

## 💡 Example Session

```bash
# Add a task — AI predicts priority as "high" and category as "Study"
python todo.py add "Prepare for neural networks exam" --due 2025-06-10

# Get AI-ranked recommendations for what to work on today
python todo.py recommend

# See how tasks cluster together
python todo.py cluster

# Check for overdue anomalies
python todo.py anomaly

# Predict next week's workload
python todo.py forecast
```

---

## 🗄️ Data Storage

All tasks stored in local SQLite (`tasks.db`). Schema: `id, title, description, priority, category, due_date, completed, created_at, completed_at`.

---

## 📄 License

Submitted as BYOP project — Fundamentals in AI and ML (CSA2001).

---

## 👤 Author

**Ashutosh Bhutekar**
Reg No: 25BAI11422
Course: Fundamentals in AI and ML — CSA2001
