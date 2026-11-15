"""MiniLearn backend. Generated from minilearn_backend_v2.ipynb: only the definitions, no demo code."""


import csv


import json


import os


import time


import numpy as np


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)                       # first line = column names
        rows = [row for row in reader if len(row) == len(header)]
    return header, rows


def is_missing(text):
    return text.strip().lower() in ("", "na", "nan", "null", "none", "?")


def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def column_type(values):
    present = [v for v in values if not is_missing(v)]
    if len(present) == 0:
        return "empty"
    for v in present:
        if not is_number(v):
            return "text"
    return "numeric"


def looks_like_id(present):
    numbers = [float(v) for v in present]
    all_whole = all(x.is_integer() for x in numbers)                # 1, 2, 3 and not 1.5
    all_different = len(set(numbers)) == len(numbers)               # no value repeats
    consecutive = max(numbers) - min(numbers) + 1 == len(numbers)   # 1, 2, 3 ... with no gaps
    return all_whole and all_different and consecutive


def profile_columns(header, rows):
    info = []
    for i, name in enumerate(header):
        values = [row[i] for row in rows]
        present = [v for v in values if not is_missing(v)]
        info.append({
            "name": name,
            "type": column_type(values),
            "missing": len(values) - len(present),
            "unique": len(set(present)),      # how many different values
        })
    return info


def detect_task(info, target):
    col = next(c for c in info if c["name"] == target)
    if col["type"] == "numeric" and col["unique"] > 10:
        return "regression"          # many different numbers: predict a number
    return "classification"


def prepare_data(header, rows, target, task):
    t = header.index(target)
    rows = [r for r in rows if not is_missing(r[t])]    # a row needs a target

    # the target column becomes y
    if task == "regression":
        y = np.array([float(r[t]) for r in rows])
        labels = None
    else:
        values = {r[t] for r in rows}
        all_numbers = all(is_number(v) for v in values)
        labels = sorted(values, key=float if all_numbers else str)
        y = np.array([labels.index(r[t]) for r in rows])

    # the other columns become X
    columns, names = [], []
    for i, name in enumerate(header):
        if i == t:
            continue
        values = [r[i] for r in rows]
        present = [v for v in values if not is_missing(v)]
        kind = column_type(values)
        if kind == "numeric" and looks_like_id(present):
            continue                                     # an ID carries no meaning, skip it
        if kind == "numeric":
            median = float(np.median([float(v) for v in present]))
            columns.append([float(v) if not is_missing(v) else median for v in values])
            names.append(name)
        elif kind == "text" and len(set(present)) <= 10:
            for category in sorted(set(present)):
                columns.append([1.0 if v == category else 0.0 for v in values])
                names.append(name + "=" + category)
        # text columns with many different values (like names) are skipped

    X = np.array(columns).T if columns else np.zeros((len(y), 0))
    return X, y, labels, names


def scale(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1                 # avoid dividing by zero
    return (train - mean) / std, (test - mean) / std


class Baseline:
    def __init__(self, task):
        self.task = task

    def fit(self, X, y):
        if self.task == "classification":
            self.value = np.bincount(y).argmax()     # most common class
        else:
            self.value = y.mean()                    # average number
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


class RidgeRegression:
    # alpha=0 is plain linear regression, alpha>0 is ridge
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xb = np.c_[np.ones(len(X)), X]                # column of 1s = the intercept
        penalty = self.alpha * np.eye(Xb.shape[1])
        penalty[0, 0] = 0                             # never shrink the intercept
        self.w = np.linalg.lstsq(Xb.T @ Xb + penalty, Xb.T @ y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.w


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.classes = np.unique(y)
        Xb = np.c_[np.ones(len(X)), X]
        self.W = np.zeros((len(self.classes), Xb.shape[1]))   # one row of weights per class
        for i, c in enumerate(self.classes):
            target = (y == c).astype(float)                   # 1 for this class, 0 for the rest
            for _ in range(self.epochs):
                z = np.clip(Xb @ self.W[i], -30, 30)
                p = 1 / (1 + np.exp(-z))                      # probability of this class
                self.W[i] -= self.lr * Xb.T @ (p - target) / len(target)
        return self

    def predict(self, X):
        Xb = np.c_[np.ones(len(X)), X]
        return self.classes[(Xb @ self.W.T).argmax(axis=1)]


class LinearSVM:
    def __init__(self, lr=0.05, epochs=200, reg=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def fit(self, X, y):
        self.classes = np.unique(y)
        Xb = np.c_[np.ones(len(X)), X]
        self.W = np.zeros((len(self.classes), Xb.shape[1]))
        for i, c in enumerate(self.classes):
            sign = np.where(y == c, 1.0, -1.0)
            for _ in range(self.epochs):
                unsure = sign * (Xb @ self.W[i]) < 1          # rows inside the margin
                grad = self.reg * self.W[i] - Xb[unsure].T @ sign[unsure] / len(y)
                self.W[i] -= self.lr * grad
        return self

    def predict(self, X):
        Xb = np.c_[np.ones(len(X)), X]
        return self.classes[(Xb @ self.W.T).argmax(axis=1)]


class KNN:
    def __init__(self, task, k=5):
        self.task = task
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        guesses = []
        for row in X:
            distances = ((self.X - row) ** 2).sum(axis=1)
            nearest = self.y[np.argsort(distances)[:self.k]]
            if self.task == "classification":
                guesses.append(np.bincount(nearest).argmax())   # the neighbours vote
            else:
                guesses.append(nearest.mean())                  # the neighbours are averaged
        return np.array(guesses)


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.vars = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes])
        self.priors = np.array([(y == c).mean() for c in self.classes])
        return self

    def predict(self, X):
        scores = []
        for i in range(len(self.classes)):
            fit_to_class = -0.5 * (np.log(2 * np.pi * self.vars[i])
                                   + (X - self.means[i]) ** 2 / self.vars[i]).sum(axis=1)
            scores.append(np.log(self.priors[i]) + fit_to_class)
        return self.classes[np.argmax(scores, axis=0)]


class DecisionTree:
    def __init__(self, task, max_depth=6, min_samples=5, max_features=None, seed=0):
        self.task = task
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y):
        self.rng = np.random.default_rng(self.seed)
        self.n_classes = int(y.max()) + 1 if self.task == "classification" else 0
        self.root = self._grow(X, y, 0)
        return self

    def _impurity(self, y):
        if self.task == "classification":
            p = np.bincount(y, minlength=self.n_classes) / len(y)
            return 1 - (p ** 2).sum()        # Gini: how mixed are the classes?
        return y.var()                       # how spread out are the numbers?

    def _leaf(self, y):
        if self.task == "classification":
            return {"value": int(np.bincount(y, minlength=self.n_classes).argmax())}
        return {"value": float(y.mean())}

    def _grow(self, X, y, depth):
        too_deep = depth >= self.max_depth
        too_small = len(y) < 2 * self.min_samples
        if too_deep or too_small or self._impurity(y) == 0:
            return self._leaf(y)
        split = self._best_split(X, y)
        if split is None:
            return self._leaf(y)
        feature, threshold = split
        left = X[:, feature] <= threshold
        return {"feature": feature, "threshold": threshold,
                "left": self._grow(X[left], y[left], depth + 1),
                "right": self._grow(X[~left], y[~left], depth + 1)}

    def _best_split(self, X, y):
        n_features = X.shape[1]
        how_many = min(self.max_features or n_features, n_features)
        features = self.rng.choice(n_features, size=how_many, replace=False)
        best_cost = self._impurity(y) * len(y)
        best = None
        for f in features:
            # try 9 cut points: the 10%, 20% ... 90% values of this feature
            for t in np.unique(np.percentile(X[:, f], range(10, 100, 10))):
                left = X[:, f] <= t
                n_left = left.sum()
                n_right = len(y) - n_left
                if n_left < self.min_samples or n_right < self.min_samples:
                    continue
                cost = self._impurity(y[left]) * n_left + self._impurity(y[~left]) * n_right
                if cost < best_cost:
                    best_cost = cost
                    best = (f, t)
        return best

    def predict(self, X):
        out = []
        for row in X:
            node = self.root
            while "value" not in node:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            out.append(node["value"])
        return np.array(out)


class RandomForest:
    def __init__(self, task, n_trees=15, max_depth=6):
        self.task = task
        self.n_trees = n_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        rng = np.random.default_rng(0)
        per_split = max(1, int(np.sqrt(X.shape[1])))
        self.trees = []
        for i in range(self.n_trees):
            sample = rng.integers(0, len(y), len(y))      # draw rows with repeats
            tree = DecisionTree(self.task, self.max_depth, max_features=per_split, seed=i)
            self.trees.append(tree.fit(X[sample], y[sample]))
        return self

    def predict(self, X):
        votes = np.array([tree.predict(X) for tree in self.trees])
        if self.task == "classification":
            return np.array([np.bincount(col.astype(int)).argmax() for col in votes.T])
        return votes.mean(axis=0)


def accuracy(y, p):
    return float((y == p).mean())


def macro_f1(y, p):
    scores = []
    for c in np.unique(y):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        scores.append(2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0)
    return float(np.mean(scores))


def r2(y, p):
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


METRICS = {
    "classification": (accuracy, "Accuracy", macro_f1, "Macro F1"),
    "regression": (r2, "R2", rmse, "RMSE"),
}


def make_folds(y, folds, rng, stratified):
    if not stratified:
        return np.array_split(rng.permutation(len(y)), folds)
    parts = [[] for _ in range(folds)]
    for c in np.unique(y):
        rows_of_class = rng.permutation(np.where(y == c)[0])
        for i, chunk in enumerate(np.array_split(rows_of_class, folds)):
            parts[i].extend(chunk.tolist())
    return [np.array(p) for p in parts]


def cross_validate(make_model, X, y, task, repeats=3, folds=5, seed=0):
    main_metric, _, extra_metric, _ = METRICS[task]
    main_scores, extra_scores, train_scores = [], [], []
    start = time.time()
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        for test in make_folds(y, folds, rng, task == "classification"):
            train = np.setdiff1d(np.arange(len(y)), test)
            X_train, X_test = scale(X[train], X[test])
            model = make_model().fit(X_train, y[train])
            guess = model.predict(X_test)
            main_scores.append(main_metric(y[test], guess))
            extra_scores.append(extra_metric(y[test], guess))
            # how well does it do on the rows it learned from? (used for the overfitting check)
            train_scores.append(main_metric(y[train], model.predict(X_train)))
    return {"score": float(np.mean(main_scores)),
            "spread": float(np.std(main_scores)),     # how much it changes between folds
            "extra": float(np.mean(extra_scores)),
            "train_score": float(np.mean(train_scores)),
            "seconds": time.time() - start}


MODELS = {
    "classification": [
        ("Baseline (most common class)", lambda: Baseline("classification")),
        ("Logistic regression",          lambda: LogisticRegression()),
        ("Linear SVM",                   lambda: LinearSVM()),
        ("KNN",                          lambda: KNN("classification")),
        ("Naive Bayes",                  lambda: GaussianNaiveBayes()),
        ("Decision tree",                lambda: DecisionTree("classification")),
        ("Random forest",                lambda: RandomForest("classification")),
    ],
    "regression": [
        ("Baseline (average value)",     lambda: Baseline("regression")),
        ("Linear regression",            lambda: RidgeRegression(0.0)),
        ("Ridge regression",             lambda: RidgeRegression(1.0)),
        ("KNN regressor",                lambda: KNN("regression")),
        ("Decision tree regressor",      lambda: DecisionTree("regression")),
        ("Random forest regressor",      lambda: RandomForest("regression")),
    ],
}


TUNING = {
    "Logistic regression": [("learning rate 0.01", lambda: LogisticRegression(lr=0.01)),
                            ("learning rate 0.1",  lambda: LogisticRegression(lr=0.1)),
                            ("learning rate 0.5",  lambda: LogisticRegression(lr=0.5))],
    "Linear SVM":          [("reg 0.001", lambda: LinearSVM(reg=0.001)),
                            ("reg 0.01",  lambda: LinearSVM(reg=0.01)),
                            ("reg 0.1",   lambda: LinearSVM(reg=0.1))],
    "KNN":                 [("k=3",  lambda: KNN("classification", k=3)),
                            ("k=5",  lambda: KNN("classification", k=5)),
                            ("k=9",  lambda: KNN("classification", k=9)),
                            ("k=15", lambda: KNN("classification", k=15))],
    "Decision tree":       [("depth 3", lambda: DecisionTree("classification", max_depth=3)),
                            ("depth 5", lambda: DecisionTree("classification", max_depth=5)),
                            ("depth 8", lambda: DecisionTree("classification", max_depth=8))],
    "Random forest":       [("depth 4", lambda: RandomForest("classification", max_depth=4)),
                            ("depth 6", lambda: RandomForest("classification", max_depth=6)),
                            ("depth 8", lambda: RandomForest("classification", max_depth=8))],
    "Ridge regression":    [("alpha 0.1", lambda: RidgeRegression(0.1)),
                            ("alpha 1",   lambda: RidgeRegression(1.0)),
                            ("alpha 10",  lambda: RidgeRegression(10.0)),
                            ("alpha 100", lambda: RidgeRegression(100.0))],
    "KNN regressor":       [("k=3",  lambda: KNN("regression", k=3)),
                            ("k=5",  lambda: KNN("regression", k=5)),
                            ("k=9",  lambda: KNN("regression", k=9)),
                            ("k=15", lambda: KNN("regression", k=15))],
    "Decision tree regressor": [("depth 3", lambda: DecisionTree("regression", max_depth=3)),
                                ("depth 5", lambda: DecisionTree("regression", max_depth=5)),
                                ("depth 8", lambda: DecisionTree("regression", max_depth=8))],
    "Random forest regressor": [("depth 4", lambda: RandomForest("regression", max_depth=4)),
                                ("depth 6", lambda: RandomForest("regression", max_depth=6)),
                                ("depth 8", lambda: RandomForest("regression", max_depth=8))],
}


LINEAR_MODELS = ["Logistic regression", "Linear SVM", "Linear regression", "Ridge regression"]


TREE_MODELS = ["Decision tree", "Random forest", "Decision tree regressor", "Random forest regressor"]


def choose_settings(models, X, y, task, tune=True):
    """For each model, try its settings on a small inner cross-validation and keep the best."""
    chosen = []
    for name, make_model in models:
        if tune and name in TUNING:
            best = None
            for label, candidate in TUNING[name]:
                score = cross_validate(candidate, X, y, task, repeats=1, folds=3, seed=100)["score"]
                if best is None or score > best[0]:
                    best = (score, label, candidate)
            chosen.append({"name": f"{name} [{best[1]}]", "base": name,
                           "setting": best[1], "make": best[2]})
        else:
            chosen.append({"name": name, "base": name, "setting": "default", "make": make_model})
    return chosen


def permutation_importance(model, X_test, y_test, metric, repeats=5):
    """Shuffle one feature at a time. The bigger the score drops, the more the model relied on it."""
    rng = np.random.default_rng(7)
    base = metric(y_test, model.predict(X_test))
    drops = []
    for j in range(X_test.shape[1]):
        scores = []
        for _ in range(repeats):
            shuffled = X_test.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            scores.append(metric(y_test, model.predict(shuffled)))
        drops.append(base - float(np.mean(scores)))
    return drops


def data_facts(X, names, y, task):
    """Plain facts about the data that the explanations will point to."""
    facts = {"rows": int(len(y)), "features": len(names)}

    # widest and narrowest feature range (0/1 columns are skipped)
    spans = {n: float(X[:, j].max() - X[:, j].min())
             for j, n in enumerate(names) if len(np.unique(X[:, j])) > 2}
    if len(spans) >= 2:
        widest = max(spans, key=spans.get)
        narrowest = min(spans, key=spans.get)
        facts["widest"] = [widest, spans[widest]]
        facts["narrowest"] = [narrowest, spans[narrowest]]

    # most correlated pair of features (dummies from the same text column are skipped)
    best_pair, best_corr = None, 0.0
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            same_source = ("=" in names[a] and "=" in names[b]
                           and names[a].split("=")[0] == names[b].split("=")[0])
            if same_source or X[:, a].std() == 0 or X[:, b].std() == 0:
                continue
            c = abs(float(np.corrcoef(X[:, a], X[:, b])[0, 1]))
            if c > best_corr:
                best_pair, best_corr = [names[a], names[b]], c
    facts["most_correlated"] = best_pair
    facts["max_correlation"] = best_corr

    if task == "classification":
        facts["largest_class_share"] = float(np.bincount(y).max() / len(y))
    return facts


def explain_models(results, facts):
    """Turn the numbers into sentences. Every sentence points to something measured on THIS data."""
    def best_of(family):
        members = [r for r in results if r["base"] in family]
        return members[0] if members else None          # results are sorted best first

    best_tree = best_of(TREE_MODELS)
    single_tree = best_of(["Decision tree", "Decision tree regressor"])
    all_notes = []
    for r in results:
        notes = []
        base = r["base"]
        chosen = "" if r["setting"] == "default" else f" ({r['setting']}, chosen by tuning)"
        if base.startswith("Baseline"):
            notes.append("Always predicts the most common class (or the average). Every other model is measured against it.")
            all_notes.append(notes)
            continue

        if base in ("KNN", "KNN regressor"):
            notes.append(f"Looks at the closest rows and lets them vote or average{chosen}.")
            if "widest" in facts:
                notes.append(f"Distances mix all features, so they were scaled first. Otherwise {facts['widest'][0]} "
                             f"(range {facts['widest'][1]:.1f}) would swamp {facts['narrowest'][0]} "
                             f"(range {facts['narrowest'][1]:.1f}).")
        elif base in LINEAR_MODELS:
            notes.append(f"Fits a straight-line relationship{chosen}.")
            if best_tree is not None:
                gap = best_tree["score"] - r["score"]
                if gap > max(best_tree["spread"], r["spread"]):
                    notes.append(f"The best tree-based model scored {gap:.3f} higher, so the pattern here is not a straight line.")
                else:
                    notes.append("It scored about the same as the tree-based models, so a straight line captures most of the pattern.")
        elif base == "Naive Bayes":
            notes.append("Assumes the features are independent of each other.")
            if facts["most_correlated"] is not None and facts["max_correlation"] > 0.5:
                a, b = facts["most_correlated"]
                notes.append(f"Here {a} and {b} are correlated (r={facts['max_correlation']:.2f}), which goes against that assumption.")
            else:
                notes.append("No strongly correlated pair of features was found, so the assumption roughly holds.")
        elif base in ("Decision tree", "Decision tree regressor"):
            notes.append(f"Asks a chain of yes/no questions about the features{chosen}. It can follow curved patterns.")
        elif base in ("Random forest", "Random forest regressor"):
            notes.append(f"Averages many different trees{chosen}.")
            if single_tree is not None:
                diff = r["score"] - single_tree["score"]
                if diff > max(r["spread"], single_tree["spread"]):
                    notes.append(f"It beat a single tree by {diff:.3f}, more than the fold-to-fold noise, so averaging helped.")
                elif diff > 0:
                    notes.append(f"It scored {diff:.3f} above a single tree, but that is within the fold-to-fold noise, so averaging gave little benefit here.")
                else:
                    notes.append("It did not beat a single tree here, so averaging did not help on this data.")

        gap = r["train_score"] - r["score"]
        seen = f"{r['train_score']:.3f} on the rows it learned from vs {r['score']:.3f} on unseen rows"
        if gap > 0.08:
            notes.append(f"Overfitting check: {seen}. A large gap: it partly memorised the data.")
        elif gap > 0.03:
            notes.append(f"Overfitting check: {seen}. A moderate gap: it fits its own rows a little better than new ones.")
        else:
            notes.append(f"Overfitting check: {seen}. A small gap.")
        if r["tie"]:
            notes.append("Effectively tied with the best: the difference is smaller than the fold-to-fold spread.")
        all_notes.append(notes)
    return all_notes


def rank_models(X, y, task, names=None, tune=True):
    if names is None:
        names = [f"feature {j}" for j in range(X.shape[1])]
    main_metric, score_name, extra_metric, extra_name = METRICS[task]

    # 1. lock away 20% of the rows, untouched until the very end
    rng = np.random.default_rng(42)
    locked = make_folds(y, 5, rng, task == "classification")[0]
    dev = np.setdiff1d(np.arange(len(y)), locked)

    # 2. choose the best settings for each model, using only the other 80%
    chosen = choose_settings(MODELS[task], X[dev], y[dev], task, tune)

    # 3. cross-validate every model on that 80%
    results = []
    for item in chosen:
        result = cross_validate(item["make"], X[dev], y[dev], task)
        result.update(name=item["name"], base=item["base"], setting=item["setting"], make=item["make"])
        results.append(result)
    results.sort(key=lambda r: r["score"], reverse=True)

    # 4. compare with the baseline and spot ties
    baseline_score = next(r["score"] for r in results if r["base"].startswith("Baseline"))
    best = results[0]
    for r in results:
        r["vs_baseline"] = r["score"] - baseline_score
        gap = best["score"] - r["score"]
        r["tie"] = r is not best and gap < max(best["spread"], r["spread"])

    # 5. one final check of the winner on the locked rows, plus what it relied on
    X_train, X_test = scale(X[dev], X[locked])
    best_model = best["make"]().fit(X_train, y[dev])
    final_score = main_metric(y[locked], best_model.predict(X_test))
    drops = permutation_importance(best_model, X_test, y[locked], main_metric)
    top = np.argsort(drops)[::-1][:5]
    importance = [[names[j], float(drops[j])] for j in top]

    # 6. explanations
    facts = data_facts(X, names, y, task)
    notes = explain_models(results, facts)

    # 7. a clean report (plain numbers and text only, so it can be saved as JSON)
    return {
        "task": task, "rows": int(len(y)), "tuned": tune,
        "score_name": score_name, "extra_name": extra_name,
        "best": best["name"],
        "results": [{"name": r["name"], "score": r["score"], "spread": r["spread"],
                     "extra": r["extra"], "train_score": r["train_score"],
                     "seconds": r["seconds"], "vs_baseline": r["vs_baseline"],
                     "tie": bool(r["tie"]), "notes": n}
                    for r, n in zip(results, notes)],
        "final_check": {"model": best["name"], "score": final_score, "cv_score": best["score"]},
        "importance": importance,
        "facts": facts,
    }


def print_report(report):
    print(f"Task: {report['task']}   (rows used: {report['rows']})")
    print("Settings:", "tuned on the 80% used for ranking" if report["tuned"] else "default (no tuning)")
    print()
    print(f"{'Model':38}{report['score_name']:>9}{'  +/-':>8}{report['extra_name']:>10}{'  vs baseline':>14}   note")
    for r in report["results"]:
        note = "BEST" if r["name"] == report["best"] else ("tied with best" if r["tie"] else "")
        print(f"{r['name']:38}{r['score']:9.3f}{r['spread']:8.3f}{r['extra']:10.3f}{r['vs_baseline']:14.3f}   {note}")

    print("\nWhy each model ranked where it did")
    for r in report["results"]:
        print(f"\n{r['name']}")
        for line in r["notes"]:
            print("  -", line)

    print("\nWhat the best model relied on most (how far its score drops when that feature is shuffled):")
    for name, drop in report["importance"]:
        print(f"  {name:22}{drop:8.3f}")
    if any("=" in name for name, _ in report["importance"]):
        print("  (features made from the same text column share the credit between them)")

    fc = report["final_check"]
    print(f"\nFinal check: {fc['model']} scored {fc['score']:.3f} on rows it never saw "
          f"(cross-validation said {fc['cv_score']:.3f})")


def save_report(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved", path)


def run_minilearn(path, target, tune=True, save_json=None):
    if not os.path.exists(path):
        print(f"Could not find '{path}'. Put it in the same folder as this notebook.")
        return None
    header, rows = load_csv(path)
    if target not in header:
        print(f"'{target}' is not a column. The columns are: {header}")
        return None

    info = profile_columns(header, rows)
    task = detect_task(info, target)
    X, y, labels, names = prepare_data(header, rows, target, task)
    if len(names) == 0:
        print("No usable feature columns were found.")
        return None
    if len(y) < 30:
        print(f"Only {len(y)} rows have a target value. At least 30 are needed for a fair comparison.")
        return None
    if task == "classification" and len(set(y)) < 2:
        print("The target column has only one distinct value, so there is nothing to predict.")
        return None

    print("Features used:", names, "\n")
    report = rank_models(X, y, task, names, tune)
    print_report(report)
    if save_json:
        save_report(report, save_json)
    return report
