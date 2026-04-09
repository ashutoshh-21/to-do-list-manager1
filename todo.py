#!/usr/bin/env python3
"""
AI-Powered ToDo List Manager — CLI Application
Applies AI/ML algorithms: NLP text classification, priority prediction,
clustering, anomaly detection, and smart recommendations.
Course: Fundamentals in AI and ML (CSA2001)
"""

import json, os, sys, sqlite3, re, math, random
from datetime import datetime, date, timedelta
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.db")

PRIORITIES   = {"high": "🔴", "medium": "🟡", "low": "🟢"}
CATEGORIES   = ["Work", "Study", "Personal", "Health", "Finance", "College", "General"]

COLORS = {
    "red":"\033[91m","yellow":"\033[93m","green":"\033[92m",
    "cyan":"\033[96m","bold":"\033[1m","reset":"\033[0m",
    "dim":"\033[2m","blue":"\033[94m","magenta":"\033[95m","white":"\033[97m",
}

def c(text, *styles):
    codes = "".join(COLORS.get(s,"") for s in styles)
    return f"{codes}{text}{COLORS['reset']}"

# ══════════════════════════════════════════════════════════════
# ██  AI / ML ENGINE  ████████████████████████████████████████
# ══════════════════════════════════════════════════════════════

class NaiveBayesPriorityClassifier:
    """
    AI MODULE 1: Multinomial Naïve Bayes Classifier
    ─────────────────────────────────────────────────
    Learns from past tasks to predict the priority (high/medium/low)
    of a new task based on the words in its title and description.

    Algorithm: P(priority | words) ∝ P(priority) × Π P(word | priority)
    Uses Laplace smoothing to handle unseen words (add-1 smoothing).
    """

    def __init__(self):
        self.class_counts:  Dict[str, int]           = defaultdict(int)
        self.word_counts:   Dict[str, Dict[str,int]] = defaultdict(lambda: defaultdict(int))
        self.vocab:         set                       = set()
        self.total_docs:    int                       = 0
        self.trained:       bool                      = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer, removes punctuation."""
        return re.findall(r'[a-z]+', text.lower())

    def train(self, tasks: List[dict]):
        """Fit the classifier on a list of task dicts with 'title','description','priority'."""
        self.class_counts  = defaultdict(int)
        self.word_counts   = defaultdict(lambda: defaultdict(int))
        self.vocab         = set()
        self.total_docs    = 0

        for task in tasks:
            label  = task.get("priority", "medium")
            tokens = self._tokenize(task.get("title","") + " " + task.get("description",""))
            self.class_counts[label] += 1
            self.total_docs          += 1
            for tok in tokens:
                self.word_counts[label][tok] += 1
                self.vocab.add(tok)

        self.trained = len(tasks) >= 3

    def predict(self, title: str, description: str = "") -> Tuple[str, Dict[str,float]]:
        """Return (predicted_priority, {priority: probability})."""
        if not self.trained:
            return "medium", {}

        tokens  = self._tokenize(title + " " + description)
        V       = len(self.vocab)
        scores  = {}

        for label in ["high", "medium", "low"]:
            # log P(class)
            prior      = math.log((self.class_counts[label] + 1) /
                                  (self.total_docs + len(PRIORITIES)))
            likelihood = 0.0
            total_w    = sum(self.word_counts[label].values())

            for tok in tokens:
                # Laplace (add-1) smoothing
                p_w = (self.word_counts[label][tok] + 1) / (total_w + V + 1)
                likelihood += math.log(p_w)

            scores[label] = prior + likelihood

        # Convert log-probs to probabilities via softmax
        max_s  = max(scores.values())
        exp_s  = {k: math.exp(v - max_s) for k, v in scores.items()}
        total  = sum(exp_s.values())
        probs  = {k: round(v / total, 3) for k, v in exp_s.items()}
        best   = max(probs, key=probs.get)
        return best, probs


class TFIDFCategoryClassifier:
    """
    AI MODULE 2: TF-IDF + Cosine Similarity Category Classifier
    ─────────────────────────────────────────────────────────────
    Represents each category as a TF-IDF weighted centroid vector
    built from past tasks. A new task is classified by finding the
    category whose centroid has highest cosine similarity to the
    task's TF-IDF vector.

    TF(t,d)  = count(t in d) / len(d)
    IDF(t)   = log(N / df(t) + 1)
    TF-IDF   = TF × IDF
    sim(A,B) = (A·B) / (|A||B|)
    """

    def __init__(self):
        self.idf:       Dict[str, float]            = {}
        self.centroids: Dict[str, Dict[str,float]]  = {}
        self.trained:   bool                        = False

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-z]+', text.lower())

    def _tf(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        n      = max(len(tokens), 1)
        return {w: cnt / n for w, cnt in counts.items()}

    def _tfidf_vec(self, tokens: List[str]) -> Dict[str, float]:
        tf  = self._tf(tokens)
        return {w: tf[w] * self.idf.get(w, 0.0) for w in tf}

    def _cosine(self, a: Dict[str,float], b: Dict[str,float]) -> float:
        keys  = set(a) & set(b)
        dot   = sum(a[k] * b[k] for k in keys)
        mag_a = math.sqrt(sum(v**2 for v in a.values()))
        mag_b = math.sqrt(sum(v**2 for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def train(self, tasks: List[dict]):
        docs_by_cat: Dict[str, List[List[str]]] = defaultdict(list)
        all_tokens:  List[List[str]]            = []

        for task in tasks:
            tokens = self._tokenize(task.get("title","") + " " + task.get("description",""))
            cat    = task.get("category", "General")
            docs_by_cat[cat].append(tokens)
            all_tokens.append(tokens)

        # Compute IDF across all documents
        N       = max(len(all_tokens), 1)
        df      = defaultdict(int)
        for doc in all_tokens:
            for w in set(doc):
                df[w] += 1
        self.idf = {w: math.log(N / (cnt + 1)) + 1 for w, cnt in df.items()}

        # Build per-category centroid as average TF-IDF vector
        self.centroids = {}
        for cat, docs in docs_by_cat.items():
            if not docs:
                continue
            combined: Dict[str, float] = defaultdict(float)
            for doc in docs:
                for w, v in self._tfidf_vec(doc).items():
                    combined[w] += v
            n = len(docs)
            self.centroids[cat] = {w: v/n for w, v in combined.items()}

        self.trained = len(tasks) >= 3

    def predict(self, title: str, description: str = "") -> Tuple[str, Dict[str,float]]:
        if not self.trained or not self.centroids:
            return "General", {}
        tokens  = self._tokenize(title + " " + description)
        vec     = self._tfidf_vec(tokens)
        sims    = {cat: round(self._cosine(vec, centroid), 3)
                   for cat, centroid in self.centroids.items()}
        best    = max(sims, key=sims.get) if sims else "General"
        return best, sims


class KMeansTaskClusterer:
    """
    AI MODULE 3: K-Means Clustering
    ─────────────────────────────────
    Groups tasks into K clusters based on their feature vectors:
    [priority_score, days_until_due, completed_flag, title_length].

    Algorithm:
      1. Initialise K centroids randomly
      2. Assign each point to nearest centroid (Euclidean distance)
      3. Recompute centroids as cluster means
      4. Repeat until convergence (max_iter reached or no change)
    """

    def __init__(self, k: int = 3, max_iter: int = 100, seed: int = 42):
        self.k        = k
        self.max_iter = max_iter
        self.seed     = seed
        self.centroids: List[List[float]] = []
        self.labels:    List[int]         = []

    def _featurize(self, task: dict) -> List[float]:
        pri_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
        pri     = pri_map.get(task.get("priority","medium"), 2.0)
        today   = date.today()
        due     = task.get("due_date")
        if due:
            try:
                days = (datetime.strptime(due, "%Y-%m-%d").date() - today).days
                days = max(-30, min(days, 30))   # clip to [-30, 30]
            except Exception:
                days = 7.0
        else:
            days = 7.0
        completed = float(task.get("completed", 0))
        tlen      = min(len(task.get("title","")), 100) / 100.0
        return [pri / 3.0, (days + 30) / 60.0, completed, tlen]

    def _euclidean(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

    def fit(self, tasks: List[dict]) -> List[int]:
        if len(tasks) < self.k:
            return list(range(len(tasks)))

        vectors = [self._featurize(t) for t in tasks]
        random.seed(self.seed)
        self.centroids = random.sample(vectors, self.k)

        for _ in range(self.max_iter):
            # Assignment step
            labels = []
            for v in vectors:
                dists  = [self._euclidean(v, c) for c in self.centroids]
                labels.append(dists.index(min(dists)))

            # Update step
            new_centroids = []
            for kid in range(self.k):
                pts = [vectors[i] for i,l in enumerate(labels) if l == kid]
                if pts:
                    dim = len(pts[0])
                    new_centroids.append([sum(p[d] for p in pts)/len(pts) for d in range(dim)])
                else:
                    new_centroids.append(self.centroids[kid])

            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids

        self.labels = labels
        return labels

    def describe_cluster(self, cluster_id: int, tasks: List[dict]) -> str:
        member_tasks = [t for t, l in zip(tasks, self.labels) if l == cluster_id]
        if not member_tasks:
            return "Empty cluster"
        pris  = Counter(t.get("priority","medium") for t in member_tasks)
        cats  = Counter(t.get("category","General") for t in member_tasks)
        top_p = pris.most_common(1)[0][0]
        top_c = cats.most_common(1)[0][0]
        return f"{len(member_tasks)} tasks | dominant priority: {top_p} | dominant category: {top_c}"


class AnomalyDetector:
    """
    AI MODULE 4: Z-Score Anomaly Detection
    ────────────────────────────────────────
    Detects statistically unusual productivity patterns.
    Computes the mean (μ) and standard deviation (σ) of daily
    task completion counts, then flags days where:
        |count - μ| / σ  >  threshold

    Also detects tasks overdue by more than 2× the average overdue gap.
    """

    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold

    def _zscore(self, values: List[float]) -> List[float]:
        if len(values) < 2:
            return [0.0] * len(values)
        mu    = sum(values) / len(values)
        var   = sum((v - mu)**2 for v in values) / len(values)
        sigma = math.sqrt(var) if var > 0 else 1e-9
        return [(v - mu) / sigma for v in values]

    def detect_overdue_anomalies(self, tasks: List[dict]) -> List[dict]:
        """Return tasks that are anomalously overdue."""
        today  = date.today()
        delays = []
        tagged = []
        for t in tasks:
            if t.get("completed") or not t.get("due_date"):
                continue
            try:
                gap = (today - datetime.strptime(t["due_date"],"%Y-%m-%d").date()).days
            except Exception:
                gap = 0
            if gap > 0:
                delays.append(gap)
                tagged.append((t, gap))

        if not delays or len(delays) < 2:
            return [t for t,_ in tagged]

        mu    = sum(delays) / len(delays)
        var   = sum((d-mu)**2 for d in delays) / len(delays)
        sigma = math.sqrt(var) if var > 0 else 1.0
        return [t for t, gap in tagged if (gap - mu) / sigma > self.z_threshold * 0.5]

    def detect_completion_bursts(self, tasks: List[dict]) -> List[Tuple[str,int,float]]:
        """Flag days with anomalously high or low completion counts."""
        completions: Dict[str,int] = defaultdict(int)
        for t in tasks:
            if t.get("completed") and t.get("completed_at"):
                day = t["completed_at"][:10]
                completions[day] += 1

        if len(completions) < 3:
            return []

        days   = sorted(completions.keys())
        counts = [completions[d] for d in days]
        zs     = self._zscore(counts)
        return [(days[i], counts[i], round(zs[i], 2))
                for i in range(len(days)) if abs(zs[i]) > self.z_threshold]


class LinearRegressionDueDatePredictor:
    """
    AI MODULE 5: Simple Linear Regression — Workload Forecasting
    ──────────────────────────────────────────────────────────────
    Fits a line  y = w*x + b  via the closed-form OLS solution:
        w = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        b = ȳ - w * x̄

    x = day index (0,1,2,…)
    y = number of tasks added on that day

    Predicts the expected number of new tasks for the next N days,
    helping users plan workload.
    """

    def __init__(self):
        self.w = 0.0
        self.b = 0.0
        self.trained = False

    def fit(self, tasks: List[dict]):
        daily: Dict[str,int] = defaultdict(int)
        for t in tasks:
            day = t.get("created_at","")[:10]
            if day:
                daily[day] += 1

        if len(daily) < 3:
            return

        days   = sorted(daily.keys())
        xs     = list(range(len(days)))
        ys     = [daily[d] for d in days]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        num    = sum((xs[i]-x_mean)*(ys[i]-y_mean) for i in range(len(xs)))
        den    = sum((xs[i]-x_mean)**2 for i in range(len(xs)))
        self.w = num / den if den != 0 else 0.0
        self.b = y_mean - self.w * x_mean
        self.trained = True
        self._n = len(days)

    def predict_next(self, horizon: int = 7) -> List[Tuple[str, float]]:
        if not self.trained:
            return []
        today    = date.today()
        results  = []
        for i in range(1, horizon + 1):
            x    = self._n + i - 1
            pred = max(0.0, round(self.w * x + self.b, 2))
            day  = (today + timedelta(days=i)).isoformat()
            results.append((day, pred))
        return results


class SmartRecommender:
    """
    AI MODULE 6: Rule-Based + Frequency Scoring Recommendation Engine
    ──────────────────────────────────────────────────────────────────
    Generates personalised task recommendations using:
    - Frequency analysis of past categories and priorities
    - Time-pattern mining (which days user is most productive)
    - Urgency scoring:  score = priority_weight / (days_remaining + 1)
    - Collaborative rule mining: if high-priority tasks cluster on
      certain days, recommend scheduling new high-priority tasks there
    """

    def __init__(self):
        self.category_freq: Counter = Counter()
        self.priority_freq: Counter = Counter()
        self.peak_days:    List[str] = []

    def fit(self, tasks: List[dict]):
        completions_by_day: Dict[str,int] = defaultdict(int)
        for t in tasks:
            self.category_freq[t.get("category","General")] += 1
            self.priority_freq[t.get("priority","medium")]   += 1
            if t.get("completed") and t.get("completed_at"):
                dow = datetime.strptime(t["completed_at"][:10], "%Y-%m-%d").strftime("%A")
                completions_by_day[dow] += 1

        if completions_by_day:
            max_count = max(completions_by_day.values())
            self.peak_days = [d for d, c in completions_by_day.items() if c >= max_count * 0.75]

    def urgency_score(self, task: dict) -> float:
        pri_w = {"high": 3.0, "medium": 2.0, "low": 1.0}
        w     = pri_w.get(task.get("priority","medium"), 2.0)
        due   = task.get("due_date")
        if due:
            try:
                days = (datetime.strptime(due,"%Y-%m-%d").date() - date.today()).days
                days = max(days, 0)
            except Exception:
                days = 7
        else:
            days = 7
        return round(w / (days + 1), 4)

    def recommend(self, pending_tasks: List[dict]) -> List[dict]:
        """Return top-5 tasks sorted by urgency score with advice."""
        if not pending_tasks:
            return []
        scored = [(t, self.urgency_score(t)) for t in pending_tasks]
        scored.sort(key=lambda x: x[1], reverse=True)
        top5   = scored[:5]
        today_dow = datetime.now().strftime("%A")
        recs = []
        for task, score in top5:
            advice = ""
            if task.get("priority") == "high" and today_dow in self.peak_days:
                advice = "⭐ Great day to tackle this — matches your peak productivity!"
            elif task.get("due_date"):
                try:
                    days_left = (datetime.strptime(task["due_date"],"%Y-%m-%d").date() - date.today()).days
                    if days_left <= 1:
                        advice = "🚨 Due very soon — prioritise today!"
                    elif days_left <= 3:
                        advice = "⚠️  Due within 3 days — start soon."
                    else:
                        advice = f"📅 {days_left} days remaining."
                except Exception:
                    pass
            else:
                advice = "📌 No due date — consider setting one."
            recs.append({"task": task, "score": score, "advice": advice})
        return recs


# ══════════════════════════════════════════════════════════════
# ██  DATABASE LAYER  ████████████████████████████████████████
# ══════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT    NOT NULL,
            description  TEXT    DEFAULT '',
            priority     TEXT    DEFAULT 'medium',
            category     TEXT    DEFAULT 'General',
            due_date     TEXT    DEFAULT NULL,
            completed    INTEGER DEFAULT 0,
            created_at   TEXT    NOT NULL,
            completed_at TEXT    DEFAULT NULL
        )
    """)
    conn.commit(); conn.close()

def get_conn(): return sqlite3.connect(DB_PATH)

def fetch_all_tasks() -> List[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,title,description,priority,category,due_date,completed,created_at,completed_at FROM tasks")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols,r)) for r in cur.fetchall()]

def fetch_pending_tasks() -> List[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,title,description,priority,category,due_date,completed,created_at,completed_at FROM tasks WHERE completed=0")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols,r)) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════
# ██  CORE OPERATIONS  ███████████████████████████████████████
# ══════════════════════════════════════════════════════════════

_nb_clf   = NaiveBayesPriorityClassifier()
_tfidf_clf = TFIDFCategoryClassifier()
_reg      = LinearRegressionDueDatePredictor()
_rec      = SmartRecommender()

def _retrain():
    """Retrain all models on current DB data."""
    tasks = fetch_all_tasks()
    _nb_clf.train(tasks)
    _tfidf_clf.train(tasks)
    _reg.fit(tasks)
    _rec.fit(tasks)


def add_task(title: str, description: str = "", priority: str = "",
             category: str = "", due_date: Optional[str] = None,
             auto_predict: bool = True):
    if not title.strip():
        print(c("Error: Task title cannot be empty.", "red")); return None

    _retrain()

    # ── AI: Priority prediction (Naïve Bayes) ──
    predicted_priority, priority_probs = _nb_clf.predict(title, description)
    if not priority:
        priority = predicted_priority
        if priority_probs:
            print(c(f"\n  🤖 AI Priority Prediction  (Naïve Bayes):", "magenta", "bold"))
            for p, prob in sorted(priority_probs.items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                print(f"     {PRIORITIES[p]} {p:<8} {bar:<20} {prob*100:.1f}%")
            print(c(f"  → Auto-assigned priority: {c(priority,'bold')}", "magenta"))

    if priority not in PRIORITIES:
        print(c(f"Error: Priority must be one of {list(PRIORITIES.keys())}.", "red")); return None

    # ── AI: Category prediction (TF-IDF + Cosine Similarity) ──
    predicted_category, cat_sims = _tfidf_clf.predict(title, description)
    if not category and _tfidf_clf.trained:
        category = predicted_category
        print(c(f"\n  🤖 AI Category Suggestion  (TF-IDF Cosine Sim):", "cyan", "bold"))
        top3 = sorted(cat_sims.items(), key=lambda x: -x[1])[:3]
        for cat, sim in top3:
            bar = "█" * int(sim * 30)
            print(f"     {cat:<15} {bar:<30} sim={sim:.3f}")
        print(c(f"  → Auto-assigned category: {c(category,'bold')}", "cyan"))
    elif not category:
        category = "General"

    if due_date:
        try: datetime.strptime(due_date, "%Y-%m-%d")
        except ValueError:
            print(c("Error: Due date must be YYYY-MM-DD.", "red")); return None

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tasks (title,description,priority,category,due_date,created_at) VALUES (?,?,?,?,?,?)",
            (title.strip(), description.strip(), priority, category.strip(), due_date, created_at))
        task_id = cur.lastrowid; conn.commit()

    print(c(f"\n  ✅ Task #{task_id} added successfully!", "green", "bold"))
    return task_id


def list_tasks(filter_status="pending", filter_priority=None,
               filter_category=None, search=None):
    query  = "SELECT id,title,description,priority,category,due_date,completed,created_at FROM tasks WHERE 1=1"
    params = []
    if filter_status == "pending": query += " AND completed=0"
    elif filter_status == "done":  query += " AND completed=1"
    if filter_priority: query += " AND priority=?"; params.append(filter_priority)
    if filter_category: query += " AND LOWER(category)=LOWER(?)"; params.append(filter_category)
    if search:
        query += " AND (LOWER(title) LIKE LOWER(?) OR LOWER(description) LIKE LOWER(?))"
        params.extend([f"%{search}%", f"%{search}%"])
    query += " ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, id"

    with get_conn() as conn:
        cur = conn.cursor(); cur.execute(query, params); rows = cur.fetchall()

    if not rows:
        print(c("\n  No tasks found.", "dim")); return

    today = date.today()
    print()
    print(c(f"  {'ID':<5} {'Title':<28} {'Priority':<10} {'Category':<14} {'Due Date':<12} Status", "bold"))
    print(c("  " + "─"*83, "dim"))
    for row in rows:
        tid, title, desc, priority, category, due_date, completed, created_at = row
        pri_icon  = PRIORITIES.get(priority,"⬜")
        status    = c("✓ Done","green") if completed else c("○ Pending","yellow")
        t_display = (title[:25]+"…") if len(title)>28 else title
        due_display = ""
        if due_date:
            due_obj     = datetime.strptime(due_date,"%Y-%m-%d").date()
            overdue     = not completed and due_obj < today
            due_display = c(due_date,"red") if overdue else due_date
        else:
            due_display = c("None","dim")
        pri_color = {"high":"red","medium":"yellow","low":"green"}.get(priority,"reset")
        pri_disp  = c(f"{pri_icon} {priority}", pri_color)
        print(f"  {tid:<5} {t_display:<28} {pri_disp:<20} {category:<14} {due_display:<20} {status}")
    print()


def complete_task(task_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,title,completed FROM tasks WHERE id=?", (task_id,))
        row = cur.fetchone()
        if not row:    print(c(f"Error: Task #{task_id} not found.","red")); return
        if row[2]:     print(c(f"Task #{task_id} already completed.","yellow")); return
        completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("UPDATE tasks SET completed=1, completed_at=? WHERE id=?", (completed_at, task_id))
        conn.commit()
    print(c(f"\n  🎉 Task #{task_id} '{row[1]}' marked complete!", "green","bold"))


def delete_task(task_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT title FROM tasks WHERE id=?", (task_id,))
        row = cur.fetchone()
        if not row: print(c(f"Error: Task #{task_id} not found.","red")); return
        confirm = input(c(f"  Delete '{row[0]}'? (yes/no): ","yellow")).strip().lower()
        if confirm in ("yes","y"):
            cur.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            conn.commit()
            print(c(f"\n  🗑  Task #{task_id} deleted.","red"))
        else:
            print(c("  Deletion cancelled.","dim"))


def update_task(task_id: int, **kwargs):
    allowed = {"title","description","priority","category","due_date"}
    updates = {k:v for k,v in kwargs.items() if k in allowed and v is not None}
    if not updates: print(c("Nothing to update.","dim")); return
    if "priority" in updates and updates["priority"] not in PRIORITIES:
        print(c(f"Error: Priority must be one of {list(PRIORITIES.keys())}.","red")); return
    if "due_date" in updates and updates["due_date"]:
        try: datetime.strptime(updates["due_date"],"%Y-%m-%d")
        except ValueError: print(c("Error: Date must be YYYY-MM-DD.","red")); return
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values     = list(updates.values()) + [task_id]
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT id FROM tasks WHERE id=?", (task_id,))
        if not cur.fetchone(): print(c(f"Error: Task #{task_id} not found.","red")); return
        cur.execute(f"UPDATE tasks SET {set_clause} WHERE id=?", values)
        conn.commit()
    print(c(f"\n  ✏️  Task #{task_id} updated!", "cyan","bold"))


def view_task(task_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
        row = cur.fetchone()
    if not row: print(c(f"Error: Task #{task_id} not found.","red")); return
    tid,title,description,priority,category,due_date,completed,created_at,completed_at = row
    overdue = ""
    if due_date and not completed:
        if datetime.strptime(due_date,"%Y-%m-%d").date() < date.today():
            overdue = c("  ⚠️  OVERDUE!","red","bold")
    print()
    print(c("  ┌── Task Details ──────────────────────────","cyan"))
    print(c("  │  ID       : ","dim") + c(str(tid),"bold"))
    print(c("  │  Title    : ","dim") + c(title,"bold"))
    print(c("  │  Desc     : ","dim") + (description or c("(none)","dim")))
    print(c("  │  Priority : ","dim") + c(f"{PRIORITIES[priority]} {priority}",{"high":"red","medium":"yellow","low":"green"}[priority]))
    print(c("  │  Category : ","dim") + category)
    print(c("  │  Due Date : ","dim") + (due_date or c("Not set","dim")) + overdue)
    print(c("  │  Status   : ","dim") + (c("✓ Completed","green") if completed else c("○ Pending","yellow")))
    print(c("  │  Created  : ","dim") + created_at)
    if completed_at: print(c("  │  Finished : ","dim") + completed_at)
    print(c("  └──────────────────────────────────────────","cyan"))
    print()


def show_stats():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM tasks");            total   = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM tasks WHERE completed=1"); done = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM tasks WHERE completed=0"); pending = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM tasks WHERE completed=0 AND due_date < date('now')"); overdue = cur.fetchone()[0]
        cur.execute("SELECT priority,COUNT(*) FROM tasks WHERE completed=0 GROUP BY priority"); by_pri = dict(cur.fetchall())
        cur.execute("SELECT category,COUNT(*) FROM tasks GROUP BY category ORDER BY COUNT(*) DESC LIMIT 5"); by_cat = cur.fetchall()

    pct    = int((done/total*100) if total else 0)
    filled = int(30 * pct / 100)
    bar    = c("█"*filled,"green") + c("░"*(30-filled),"dim")

    print()
    print(c("  ╔══ Task Statistics ═══════════════════════╗","cyan","bold"))
    print(c(f"  ║  Total Tasks    : {total:<24}║","cyan"))
    print(c(f"  ║  Completed      : {done:<24}║","cyan"))
    print(c(f"  ║  Pending        : {pending:<24}║","cyan"))
    print(c(f"  ║  Overdue        : {overdue:<24}║","cyan"))
    print(c(f"  ╠══ Completion Progress ════════════════════╣","cyan","bold"))
    print(f"  {c('║','cyan')}  {bar}  {c(str(pct)+'%','bold')}           {c('║','cyan')}")
    print(c(f"  ╠══ Pending by Priority ════════════════════╣","cyan","bold"))
    for pri,icon in PRIORITIES.items():
        cnt = by_pri.get(pri,0)
        print(c(f"  ║  {icon} {pri:<10}: {cnt:<24}║","cyan"))
    print(c(f"  ╠══ Top Categories ═════════════════════════╣","cyan","bold"))
    for cat, cnt in by_cat:
        line = f"  ║  {cat:<15}: {cnt:<24}║"
        print(c(line,"cyan"))
    print(c("  ╚══════════════════════════════════════════╝","cyan","bold"))
    print()


def export_tasks(filepath: str):
    tasks = fetch_all_tasks()
    with open(filepath,"w") as f: json.dump(tasks, f, indent=2)
    print(c(f"\n  📤 {len(tasks)} tasks exported to {filepath}","green"))


def import_tasks(filepath: str):
    if not os.path.exists(filepath):
        print(c(f"Error: File '{filepath}' not found.","red")); return
    with open(filepath) as f: rows = json.load(f)
    count = 0
    with get_conn() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                "INSERT INTO tasks (title,description,priority,category,due_date,completed,created_at,completed_at) VALUES (?,?,?,?,?,?,?,?)",
                (r.get("title",""), r.get("description",""), r.get("priority","medium"),
                 r.get("category","General"), r.get("due_date"), r.get("completed",0),
                 r.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), r.get("completed_at")))
            count += 1
        conn.commit()
    print(c(f"\n  📥 {count} tasks imported!", "green"))


# ══════════════════════════════════════════════════════════════
# ██  AI FEATURE SCREENS  ████████████████████████████████████
# ══════════════════════════════════════════════════════════════

def show_smart_recommendations():
    """AI MODULE 6: Smart task recommendations by urgency score."""
    _retrain()
    pending = fetch_pending_tasks()
    recs    = _rec.recommend(pending)
    if not recs:
        print(c("\n  No pending tasks to recommend.", "dim")); return

    print(c("\n  ╔══ 🤖 AI Smart Recommendations ══════════════════════╗","magenta","bold"))
    print(c(  "  ║  Algorithm: Urgency Scoring  (priority / days+1)    ║","magenta"))
    print(c(  "  ╚══════════════════════════════════════════════════════╝","magenta","bold"))
    print()
    for i, rec in enumerate(recs, 1):
        t = rec["task"]
        pri_color = {"high":"red","medium":"yellow","low":"green"}.get(t["priority"],"reset")
        print(f"  {c(str(i)+'.','bold','cyan')} {c(t['title'],'bold')}")
        print(f"     Priority : {c(PRIORITIES[t['priority']]+' '+t['priority'], pri_color)}")
        print(f"     Category : {t.get('category','General')}")
        print(f"     Due Date : {t.get('due_date') or c('Not set','dim')}")
        print(f"     Score    : {c(str(rec['score']),'yellow')}  {rec['advice']}")
        print()


def show_clustering():
    """AI MODULE 3: K-Means clustering of all tasks."""
    tasks = fetch_all_tasks()
    if len(tasks) < 3:
        print(c("\n  Need at least 3 tasks for clustering.", "dim")); return

    k          = min(3, len(tasks))
    clusterer  = KMeansTaskClusterer(k=k)
    labels     = clusterer.fit(tasks)

    print(c("\n  ╔══ 🤖 AI Task Clustering  (K-Means, K="+str(k)+") ══════════╗","cyan","bold"))
    print(c(  "  ║  Features: priority, days_until_due, status, title_len  ║","cyan"))
    print(c(  "  ╚══════════════════════════════════════════════════════════╝","cyan","bold"))
    print()

    for kid in range(k):
        desc    = clusterer.describe_cluster(kid, tasks)
        members = [(tasks[i], l) for i, l in enumerate(labels) if l == kid]
        print(c(f"  Cluster {kid+1}: {desc}", "bold"))
        for t, _ in members[:5]:
            pri_color = {"high":"red","medium":"yellow","low":"green"}.get(t["priority"],"reset")
            print(f"    • [{c(t['priority'], pri_color)}] {t['title']}")
        if len(members) > 5:
            print(c(f"    … and {len(members)-5} more","dim"))
        print()


def show_anomaly_report():
    """AI MODULE 4: Z-Score anomaly detection."""
    tasks    = fetch_all_tasks()
    detector = AnomalyDetector(z_threshold=1.5)

    print(c("\n  ╔══ 🤖 AI Anomaly Detection  (Z-Score) ════════════════╗","yellow","bold"))
    print(c(  "  ║  Flags tasks/days that deviate > 1.5σ from mean       ║","yellow"))
    print(c(  "  ╚══════════════════════════════════════════════════════╝","yellow","bold"))
    print()

    anomalous_tasks = detector.detect_overdue_anomalies(tasks)
    print(c("  📌 Anomalously Overdue Tasks:", "bold"))
    if anomalous_tasks:
        for t in anomalous_tasks:
            print(c(f"    ⚠️  #{t['id']} {t['title']}  (due: {t['due_date']})","red"))
    else:
        print(c("    None detected — great job staying on track! ✅","green"))

    print()
    bursts = detector.detect_completion_bursts(tasks)
    print(c("  📈 Anomalous Productivity Days:", "bold"))
    if bursts:
        for day, count, z in bursts:
            label = c("🔥 HIGH burst","green") if z > 0 else c("📉 LOW burst","red")
            print(f"    {label}  {day}  ({count} completions, z={z})")
    else:
        print(c("    Not enough completion history yet (need 3+ days).","dim"))
    print()


def show_workload_forecast():
    """AI MODULE 5: Linear regression workload forecast."""
    tasks = fetch_all_tasks()
    _reg.fit(tasks)
    preds = _reg.predict_next(horizon=7)

    print(c("\n  ╔══ 🤖 AI Workload Forecast  (Linear Regression) ══════╗","blue","bold"))
    print(c(  "  ║  Predicts expected new tasks per day for next 7 days   ║","blue"))
    print(c(  "  ╚══════════════════════════════════════════════════════╝","blue","bold"))
    print()

    if not preds:
        print(c("  Not enough historical data yet (need 3+ days of tasks).","dim"))
        print(); return

    max_pred = max(p for _,p in preds) if preds else 1
    for day, pred in preds:
        dow   = datetime.strptime(day, "%Y-%m-%d").strftime("%a")
        bar   = "█" * int((pred / max(max_pred, 0.1)) * 20)
        print(f"  {c(day,'bold')} ({dow})  {c(bar,'cyan'):<30} ~{pred:.1f} tasks expected")
    print()


def show_ai_predict_new():
    """Interactively predict priority + category for a new task title."""
    _retrain()
    print(c("\n  🤖 AI Task Analyzer — enter a task title to get predictions","magenta","bold"))
    title = input(c("  Title: ","blue")).strip()
    desc  = input(c("  Description (optional): ","blue")).strip()
    if not title: print(c("  No title entered.","dim")); return

    pri, pri_probs = _nb_clf.predict(title, desc)
    cat, cat_sims  = _tfidf_clf.predict(title, desc)

    print(c("\n  ── Naïve Bayes Priority Prediction ──","magenta"))
    if pri_probs:
        for p, prob in sorted(pri_probs.items(), key=lambda x:-x[1]):
            bar = "█" * int(prob*25)
            print(f"  {PRIORITIES[p]} {p:<8} {c(bar,'magenta'):<35} {prob*100:.1f}%")
    print(c(f"  → Predicted priority: {pri}","bold"))

    print(c("\n  ── TF-IDF Category Prediction ──","cyan"))
    if cat_sims:
        top3 = sorted(cat_sims.items(), key=lambda x:-x[1])[:3]
        for cat_name, sim in top3:
            bar = "█" * int(sim*30)
            print(f"  {cat_name:<15} {c(bar,'cyan'):<40} sim={sim:.3f}")
    print(c(f"  → Predicted category: {cat}","bold"))
    print()


# ══════════════════════════════════════════════════════════════
# ██  INTERACTIVE MENU  ██████████████████████████████████████
# ══════════════════════════════════════════════════════════════

def print_banner():
    print(c("""
  ╔══════════════════════════════════════════════════════╗
  ║     🤖  AI-Powered ToDo List Manager  🤖             ║
  ║     CLI Task Management  ×  ML Intelligence          ║
  ║     Course: Fundamentals in AI and ML (CSA2001)      ║
  ╚══════════════════════════════════════════════════════╝
""","cyan","bold"))


def print_menu():
    print(c("  ─── Main Menu ──────────────────────────────────────────","dim"))
    sections = [
        ("TASK OPERATIONS", [
            ("1","Add Task  (AI predicts priority + category)"),
            ("2","List Pending Tasks"),
            ("3","List All Tasks"),
            ("4","View Task Details"),
            ("5","Mark Task Complete"),
            ("6","Update Task"),
            ("7","Delete Task"),
            ("8","Search Tasks"),
        ]),
        ("AI / ML FEATURES", [
            ("9", "🤖 Smart Recommendations     (Urgency Scoring)"),
            ("10","🤖 Analyze New Task Title     (NB + TF-IDF)"),
            ("11","🤖 Task Clustering            (K-Means)"),
            ("12","🤖 Anomaly Detection          (Z-Score)"),
            ("13","🤖 Workload Forecast          (Linear Regression)"),
        ]),
        ("DATA & REPORTS", [
            ("14","Statistics Dashboard"),
            ("15","Export Tasks (JSON)"),
            ("16","Import Tasks (JSON)"),
            ("0", "Exit"),
        ]),
    ]
    for section, items in sections:
        print(c(f"\n  {section}", "bold","white"))
        for num, label in items:
            print(f"  {c(num+')', 'cyan','bold'):<18} {label}")
    print(c("\n  ────────────────────────────────────────────────────────","dim"))


def prompt(msg, default=""):
    val = input(c(f"  {msg}","blue")).strip()
    return val if val else default


def interactive():
    init_db()
    print_banner()
    while True:
        print_menu()
        choice = prompt("Enter choice: ").strip()

        if choice == "0":
            print(c("\n  👋 Goodbye!\n","cyan","bold")); sys.exit(0)

        elif choice == "1":
            print(c("\n  ── Add New Task ──","cyan","bold"))
            title = prompt("Title (required): ")
            if not title: print(c("  Title is required.","red")); continue
            desc     = prompt("Description (optional): ")
            priority = prompt("Priority [high/medium/low] (blank = AI predicts): ")
            category = prompt("Category (blank = AI predicts): ")
            due_date = prompt("Due date YYYY-MM-DD (optional): ")
            add_task(title, desc, priority or "", category or "", due_date or None)

        elif choice == "2":
            print(c("\n  ── Pending Tasks ──","cyan","bold"))
            list_tasks(filter_status="pending")

        elif choice == "3":
            print(c("\n  ── All Tasks ──","cyan","bold"))
            list_tasks(filter_status="all")

        elif choice == "4":
            tid = prompt("Task ID: ")
            if tid.isdigit(): view_task(int(tid))
            else: print(c("  Invalid ID.","red"))

        elif choice == "5":
            tid = prompt("Task ID to complete: ")
            if tid.isdigit(): complete_task(int(tid))
            else: print(c("  Invalid ID.","red"))

        elif choice == "6":
            tid = prompt("Task ID to update: ")
            if not tid.isdigit(): print(c("  Invalid ID.","red")); continue
            print(c("  Leave blank to keep current value.","dim"))
            update_task(int(tid),
                title    = prompt("New title: ") or None,
                description = prompt("New description: ") or None,
                priority = prompt("New priority: ") or None,
                category = prompt("New category: ") or None,
                due_date = prompt("New due date: ") or None)

        elif choice == "7":
            tid = prompt("Task ID to delete: ")
            if tid.isdigit(): delete_task(int(tid))
            else: print(c("  Invalid ID.","red"))

        elif choice == "8":
            kw = prompt("Search keyword: ")
            if kw:
                print(c(f"\n  ── Search: '{kw}' ──","cyan","bold"))
                list_tasks(filter_status="all", search=kw)

        elif choice == "9":  show_smart_recommendations()
        elif choice == "10": show_ai_predict_new()
        elif choice == "11": show_clustering()
        elif choice == "12": show_anomaly_report()
        elif choice == "13": show_workload_forecast()
        elif choice == "14": show_stats()

        elif choice == "15":
            fp = prompt("Export path (default: tasks_export.json): ", "tasks_export.json")
            export_tasks(fp)

        elif choice == "16":
            fp = prompt("Import file path: ")
            if fp: import_tasks(fp)

        else:
            print(c("  ⚠️  Invalid choice.","yellow"))
        print()


# ══════════════════════════════════════════════════════════════
# ██  CLI ENTRY POINT  ███████████████████████████████████████
# ══════════════════════════════════════════════════════════════

def print_help():
    print(c("""
  Usage: python todo.py [command] [options]

  Commands:
    (no args)          Interactive menu
    add  <title>       Add a task (AI predicts missing priority/category)
    list               List pending tasks
    done <id>          Mark task complete
    delete <id>        Delete a task
    recommend          Show AI smart recommendations
    forecast           Show workload forecast
    cluster            Show task clusters
    anomaly            Show anomaly report
    stats              Statistics dashboard
    help               This help message

  Options for 'add':
    --desc       "text"
    --priority   high|medium|low   (omit to let AI predict)
    --category   "name"            (omit to let AI predict)
    --due        YYYY-MM-DD

  Examples:
    python todo.py
    python todo.py add "Prepare for exam" --due 2025-06-10
    python todo.py recommend
    python todo.py forecast
""","cyan"))


def main():
    init_db()
    args = sys.argv[1:]
    if not args: interactive(); return

    cmd = args[0].lower()
    if cmd in ("help","--help","-h"):
        print_banner(); print_help()

    elif cmd == "add":
        if len(args) < 2: print(c("Error: Provide a task title.","red")); return
        title = args[1]; desc=""; priority=""; category=""; due_date=None
        i = 2
        while i < len(args):
            if args[i]=="--desc"     and i+1<len(args): desc=args[i+1];     i+=2
            elif args[i]=="--priority" and i+1<len(args): priority=args[i+1]; i+=2
            elif args[i]=="--category" and i+1<len(args): category=args[i+1]; i+=2
            elif args[i]=="--due"     and i+1<len(args): due_date=args[i+1]; i+=2
            else: i+=1
        add_task(title, desc, priority, category, due_date)

    elif cmd == "list":           list_tasks()
    elif cmd == "done":
        if len(args)<2 or not args[1].isdigit(): print(c("Provide valid ID.","red")); return
        complete_task(int(args[1]))
    elif cmd == "delete":
        if len(args)<2 or not args[1].isdigit(): print(c("Provide valid ID.","red")); return
        delete_task(int(args[1]))
    elif cmd == "recommend":      _retrain(); show_smart_recommendations()
    elif cmd == "forecast":       show_workload_forecast()
    elif cmd == "cluster":        show_clustering()
    elif cmd == "anomaly":        show_anomaly_report()
    elif cmd == "stats":          show_stats()
    else:
        print(c(f"Unknown command: '{cmd}'. Run 'python todo.py help'.","red"))


if __name__ == "__main__":
    main()
