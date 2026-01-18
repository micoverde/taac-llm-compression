# Experiment 3B: Task Classification for Ultra-Compression

## Study Design Document

**Principal Investigator**: Dr. Michael Park
**Affiliation**: ML Benchmarks & Task Classification Laboratory
**Target Venue**: NeurIPS 2026 (Datasets and Benchmarks Track)
**Date**: January 2026
**Version**: 1.0

---

## Abstract

This document presents a rigorous experimental design for classifying MBPP (Mostly Basic Python Problems) tasks according to their tolerance for aggressive prompt compression. We hypothesize that task complexity characteristics, rather than superficial features, determine compression tolerance. Our multi-method classification approach combines keyword-based heuristics, embedding-based clustering, and manual expert annotation to establish a validated taxonomy. The study aims to identify the subset of tasks tolerating ultra-compression (r <= 0.4) while maintaining acceptable code generation quality (Pass@1 >= 30%).

---

## 1. Research Questions

### RQ1: Taxonomic Structure
**What natural task categories exist within MBPP, and how do they differ in structural complexity?**

- H1.1: MBPP tasks cluster into 5-7 distinct categories based on semantic similarity
- H1.2: Categories exhibit differential complexity profiles (cyclomatic complexity, AST depth, token count)
- H1.3: Category membership is stable across multiple classification methods (kappa >= 0.7)

### RQ2: Compression Tolerance Differential
**Do different task categories exhibit statistically significant differences in compression tolerance?**

- H2.1: Task categories show heterogeneous compression tolerance (chi-square p < 0.01)
- H2.2: At least one category maintains Pass@1 >= 30% at r = 0.4 (ultra-compression)
- H2.3: At least one category requires r >= 0.6 to achieve Pass@1 >= 30%
- H2.4: The variance in compression tolerance between categories exceeds within-category variance (ANOVA F > 3.0)

### RQ3: Predictive Features
**Which task features best predict compression tolerance?**

- H3.1: Prompt length inversely correlates with compression tolerance (r < -0.3)
- H3.2: Solution cyclomatic complexity inversely correlates with compression tolerance (r < -0.4)
- H3.3: A multi-feature classifier can predict optimal compression ratio with MAE < 0.1
- H3.4: Prompt information density (unique tokens / total tokens) positively correlates with compression tolerance

---

## 2. Task Taxonomy

### 2.1 Proposed Classification Scheme

Based on preliminary analysis of MBPP (Austin et al., 2021) and established code complexity literature, we propose a six-category taxonomy:

| Category | Description | Expected n | Complexity Profile | Expected r_min |
|----------|-------------|------------|-------------------|----------------|
| **T1: String Manipulation** | Text processing, parsing, formatting | ~150 | Low (CC: 2-4) | 0.35-0.45 |
| **T2: List/Array Operations** | Collection transformations, filtering | ~180 | Low-Medium (CC: 3-5) | 0.40-0.50 |
| **T3: Arithmetic/Mathematical** | Number theory, basic computations | ~200 | Medium (CC: 4-6) | 0.45-0.55 |
| **T4: Control Flow Logic** | Conditionals, loops, state machines | ~120 | Medium-High (CC: 5-8) | 0.50-0.60 |
| **T5: Data Structure Manipulation** | Dicts, sets, tuples, nested structures | ~150 | Medium (CC: 4-7) | 0.45-0.55 |
| **T6: Algorithmic/Recursive** | DP, recursion, graph traversal | ~100 | High (CC: 6-12) | 0.55-0.70 |
| **T7: I/O & Formatting** | Type conversion, output formatting | ~74 | Low (CC: 1-3) | 0.30-0.40 |

**Note**: CC = Cyclomatic Complexity; r_min = minimum compression ratio for Pass@1 >= 30%

### 2.2 Representative Examples Per Category

#### T1: String Manipulation
```python
# MBPP Task #11: "Write a function to remove all characters except letters and numbers"
def remove_dirty_chars(str1):
    return ''.join(c for c in str1 if c.isalnum())
```
- **Characteristics**: Pattern matching, character iteration, simple transformations
- **Compression-relevant features**: Low semantic complexity, stereotypical patterns

#### T2: List/Array Operations
```python
# MBPP Task #56: "Write a function to find the maximum element in a given list"
def max_element(list1):
    return max(list1)
```
- **Characteristics**: Built-in function application, iteration patterns
- **Compression-relevant features**: High redundancy in natural language prompts

#### T3: Arithmetic/Mathematical
```python
# MBPP Task #97: "Write a function to find the nth Fibonacci number"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```
- **Characteristics**: Mathematical definitions, recurrence relations
- **Compression-relevant features**: Critical parameter names, precise constraints

#### T4: Control Flow Logic
```python
# MBPP Task #128: "Write a function to check if a year is a leap year"
def is_leap_year(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0
```
- **Characteristics**: Multi-branch conditionals, edge cases
- **Compression-relevant features**: Order-dependent logic, exception handling

#### T5: Data Structure Manipulation
```python
# MBPP Task #167: "Write a function to merge two dictionaries"
def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}
```
- **Characteristics**: Structure composition, key-value operations
- **Compression-relevant features**: Type specifications critical

#### T6: Algorithmic/Recursive
```python
# MBPP Task #234: "Write a function to find the longest common subsequence"
def lcs(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    if X[m-1] == Y[n-1]:
        return 1 + lcs(X, Y, m-1, n-1)
    return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))
```
- **Characteristics**: Recursive structure, memoization patterns
- **Compression-relevant features**: Algorithm name critical, structure preserved

#### T7: I/O & Formatting
```python
# MBPP Task #289: "Write a function to convert a tuple to a string"
def tuple_to_string(tup):
    return ''.join(str(x) for x in tup)
```
- **Characteristics**: Type conversion, format specification
- **Compression-relevant features**: Highly compressible, boilerplate-heavy prompts

---

## 3. Classification Methodology

### 3.1 Multi-Method Classification Pipeline

We employ a three-stage classification approach with cross-validation:

```
                    +-------------------+
                    |   MBPP Dataset    |
                    |   (n=974 tasks)   |
                    +-------------------+
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
    +---------------+  +-------------+  +--------------+
    | Method A:     |  | Method B:   |  | Method C:    |
    | Keyword-Based |  | Embedding   |  | Manual       |
    | Heuristic     |  | Clustering  |  | Annotation   |
    +---------------+  +-------------+  +--------------+
            |                |                |
            v                v                v
    +--------------------------------------------------+
    |        Consensus Classification (Voting)          |
    |  - Majority vote across 3 methods                 |
    |  - Conflict resolution via expert review          |
    +--------------------------------------------------+
                             |
                             v
    +--------------------------------------------------+
    |              Validated Taxonomy                   |
    |  - Inter-rater reliability: Cohen's kappa >= 0.7  |
    |  - Final category assignments                     |
    +--------------------------------------------------+
```

### 3.2 Method A: Keyword-Based Heuristic Classification

#### 3.2.1 Feature Extraction

```python
CLASSIFICATION_FEATURES = {
    "T1_string": {
        "prompt_keywords": [
            "string", "character", "substring", "reverse", "split", "join",
            "replace", "format", "concatenate", "capitalize", "lowercase",
            "uppercase", "strip", "trim", "letter", "word", "text", "char"
        ],
        "code_patterns": [
            r"\.split\(", r"\.join\(", r"\.replace\(", r"\.strip\(",
            r"\.upper\(", r"\.lower\(", r"\[::-1\]", r"\.find\(",
            r"\.startswith\(", r"\.endswith\(", r"re\.", r"\.isalpha\("
        ],
        "weight_prompt": 2.0,
        "weight_code": 3.0
    },
    "T2_list": {
        "prompt_keywords": [
            "list", "array", "element", "sort", "filter", "map", "reduce",
            "find", "search", "index", "append", "remove", "insert", "pop",
            "slice", "flatten", "unique", "duplicate", "nth", "kth"
        ],
        "code_patterns": [
            r"\.sort\(", r"sorted\(", r"\.append\(", r"\.remove\(",
            r"\.pop\(", r"\.index\(", r"filter\(", r"map\(", r"\[\d+\]",
            r"enumerate\(", r"zip\(", r"reversed\("
        ],
        "weight_prompt": 2.0,
        "weight_code": 3.0
    },
    "T3_math": {
        "prompt_keywords": [
            "prime", "factorial", "fibonacci", "gcd", "lcm", "power",
            "square", "cube", "sum", "product", "average", "mean", "median",
            "divisor", "multiple", "even", "odd", "number", "digit",
            "calculate", "compute", "arithmetic", "geometric", "modulo"
        ],
        "code_patterns": [
            r"math\.", r"%\s*==\s*0", r"\*\*", r"//\s*\d",
            r"range\(.*,.*,.*\)", r"factorial", r"sqrt\(", r"pow\("
        ],
        "weight_prompt": 2.5,
        "weight_code": 2.5
    },
    "T4_control": {
        "prompt_keywords": [
            "check", "verify", "validate", "condition", "if", "whether",
            "true", "false", "boolean", "flag", "state", "switch", "case"
        ],
        "code_patterns": [
            r"if\s+.*:", r"elif\s+", r"else:", r"while\s+.*:",
            r"for\s+.*in\s+range", r"break", r"continue", r"pass"
        ],
        "weight_prompt": 1.5,
        "weight_code": 2.5
    },
    "T5_data_struct": {
        "prompt_keywords": [
            "dictionary", "dict", "tuple", "set", "key", "value", "pair",
            "nested", "structure", "hash", "mapping", "lookup"
        ],
        "code_patterns": [
            r"dict\(", r"\{.*:.*\}", r"\.keys\(", r"\.values\(",
            r"\.items\(", r"tuple\(", r"set\(", r"\.get\(", r"\.update\("
        ],
        "weight_prompt": 2.0,
        "weight_code": 3.5
    },
    "T6_algorithmic": {
        "prompt_keywords": [
            "binary", "search", "traverse", "recursive", "dynamic",
            "programming", "graph", "tree", "node", "path", "depth",
            "breadth", "backtrack", "permutation", "combination",
            "subsequence", "longest", "shortest", "optimal", "minimum",
            "maximum", "memoization", "dp"
        ],
        "code_patterns": [
            r"def\s+\w+\([^)]*\):[^}]*\1\(",  # Recursive calls
            r"@lru_cache", r"memo\[", r"dp\[", r"visited", r"queue\.?",
            r"stack", r"heapq\.", r"collections\.deque"
        ],
        "weight_prompt": 3.0,
        "weight_code": 4.0
    },
    "T7_io": {
        "prompt_keywords": [
            "convert", "parse", "format", "print", "display", "output",
            "read", "input", "extract", "transform", "serialize"
        ],
        "code_patterns": [
            r"str\(", r"int\(", r"float\(", r"bool\(", r"\.format\(",
            r"f\"", r"print\(", r"json\.", r"\.encode\(", r"\.decode\("
        ],
        "weight_prompt": 1.5,
        "weight_code": 2.0
    }
}
```

#### 3.2.2 Scoring Algorithm

```python
def compute_category_score(task: dict, category: str) -> float:
    """Compute weighted score for a single category."""
    config = CLASSIFICATION_FEATURES[category]
    prompt = task["text"].lower()
    code = task["code"].lower()

    score = 0.0

    # Prompt keyword matching
    for keyword in config["prompt_keywords"]:
        if keyword in prompt:
            score += config["weight_prompt"]

    # Code pattern matching
    for pattern in config["code_patterns"]:
        matches = len(re.findall(pattern, code, re.IGNORECASE))
        score += matches * config["weight_code"]

    # Normalize by prompt length (longer prompts have more keyword opportunities)
    score = score / (len(prompt.split()) ** 0.5)

    return score

def classify_task_heuristic(task: dict) -> str:
    """Classify task using keyword-based heuristics."""
    scores = {cat: compute_category_score(task, cat)
              for cat in CLASSIFICATION_FEATURES}

    max_score = max(scores.values())
    if max_score < MINIMUM_CONFIDENCE_THRESHOLD:  # 2.0
        return "T0_ambiguous"

    return max(scores, key=scores.get)
```

### 3.3 Method B: Embedding-Based Clustering

#### 3.3.1 Embedding Generation

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.manifold import UMAP

# Model selection rationale:
# - code-search-net-python: trained on code understanding
# - all-mpnet-base-v2: strong general semantic similarity
EMBEDDING_MODEL = "microsoft/codebert-base"  # Primary
FALLBACK_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Secondary

def generate_embeddings(tasks: list[dict]) -> np.ndarray:
    """Generate embeddings for task prompts and code."""
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Concatenate prompt and code with separator
    texts = [
        f"{task['text']} [SEP] {task['code'][:500]}"  # Truncate long code
        for task in tasks
    ]

    embeddings = model.encode(texts,
                              batch_size=32,
                              show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings
```

#### 3.3.2 Clustering Strategy

```python
def cluster_tasks(embeddings: np.ndarray, n_clusters: int = 7) -> np.ndarray:
    """Cluster embeddings using hierarchical approach."""

    # Step 1: Dimensionality reduction for stability
    umap_reducer = UMAP(
        n_components=50,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced = umap_reducer.fit_transform(embeddings)

    # Step 2: Primary clustering with K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    primary_labels = kmeans.fit_predict(reduced)

    # Step 3: Validate with HDBSCAN (density-based)
    hdbscan = HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    density_labels = hdbscan.fit_predict(reduced)

    # Step 4: Consensus (use K-Means but flag disagreements)
    consensus_labels = primary_labels.copy()
    disagreements = (primary_labels != density_labels) & (density_labels != -1)

    return consensus_labels, disagreements
```

#### 3.3.3 Cluster Interpretation

Post-clustering, we interpret each cluster by:
1. Computing centroid-nearest examples (n=10 per cluster)
2. Extracting top TF-IDF keywords from cluster members
3. Mapping to taxonomy categories via expert review

### 3.4 Method C: Manual Expert Annotation

#### 3.4.1 Annotator Selection

- **N = 3 annotators** with >= 5 years Python experience
- **Qualification test**: Classify 50 pre-labeled tasks with >= 80% agreement
- **Compensation**: $0.50 per task (total budget: $1,461 for full dataset)

#### 3.4.2 Annotation Protocol

```markdown
## Task Classification Guidelines for Annotators

### Instructions
For each MBPP task, read the prompt and reference solution carefully.
Assign exactly ONE primary category from the taxonomy below.
If a task spans multiple categories, choose the DOMINANT one.

### Category Definitions

**T1 - String Manipulation**: Tasks primarily involving text processing,
character operations, pattern matching, or string transformations.
Examples: reverse string, count vowels, extract substring

**T2 - List/Array Operations**: Tasks primarily involving collection
manipulation, element access, sorting, or filtering.
Examples: find maximum, remove duplicates, rotate list

**T3 - Arithmetic/Mathematical**: Tasks involving numerical computation,
number theory, mathematical formulas, or sequences.
Examples: check prime, compute factorial, find GCD

**T4 - Control Flow Logic**: Tasks requiring complex conditional logic,
state management, or multi-branch decision making.
Examples: validate input, state machine, game logic

**T5 - Data Structure Manipulation**: Tasks primarily involving
dictionaries, sets, tuples, or nested data structures.
Examples: merge dicts, convert nested list, group by key

**T6 - Algorithmic/Recursive**: Tasks requiring known algorithms,
recursive solutions, dynamic programming, or graph operations.
Examples: binary search, LCS, tree traversal, path finding

**T7 - I/O & Formatting**: Tasks primarily involving type conversion,
formatting output, or data serialization.
Examples: int to string, format date, parse JSON

**T0 - Ambiguous**: Use ONLY if the task genuinely does not fit
any category. Must provide written justification.

### Quality Requirements
- Spend at least 30 seconds per task
- Review both prompt AND solution code
- Mark confidence level (High/Medium/Low) for each classification
```

#### 3.4.3 Annotation Interface

Tasks presented via custom web interface with:
- Task ID, prompt text, solution code (syntax highlighted)
- Radio button category selection
- Confidence dropdown (High/Medium/Low)
- Free-text justification field (required for T0 or Low confidence)
- Timer display (minimum 30s before submission allowed)

---

## 4. Validation Strategy

### 4.1 Inter-Rater Reliability

#### 4.1.1 Metrics

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **Cohen's Kappa (pairwise)** | >= 0.70 | Agreement between annotator pairs |
| **Fleiss' Kappa (multi-rater)** | >= 0.65 | Overall agreement across all annotators |
| **Krippendorff's Alpha** | >= 0.67 | Reliability with missing data |
| **Percent Agreement** | >= 80% | Raw agreement (supplementary) |

#### 4.1.2 Disagreement Resolution Protocol

```
Level 1: Majority Vote (2/3 agreement)
    -> Accept majority category

Level 2: Expert Arbitration (3-way disagreement)
    -> Senior researcher reviews and decides
    -> Document reasoning in disagreement log

Level 3: Task Exclusion (irreconcilable)
    -> Flag as "multi-category"
    -> Include in sensitivity analysis
    -> Exclude from primary analysis
```

### 4.2 Cross-Method Validation

#### 4.2.1 Agreement Matrix

Compute pairwise agreement between methods:

```python
def compute_method_agreement(
    heuristic_labels: np.ndarray,
    cluster_labels: np.ndarray,  # After interpretation mapping
    manual_labels: np.ndarray
) -> dict:
    """Compute agreement metrics between classification methods."""

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pairs = [
        ("heuristic", "cluster", heuristic_labels, cluster_labels),
        ("heuristic", "manual", heuristic_labels, manual_labels),
        ("cluster", "manual", cluster_labels, manual_labels)
    ]

    agreement = {}
    for name1, name2, labels1, labels2 in pairs:
        # Filter to tasks with valid labels in both
        mask = (labels1 != "T0_ambiguous") & (labels2 != "T0_ambiguous")

        agreement[f"{name1}_vs_{name2}"] = {
            "adjusted_rand": adjusted_rand_score(labels1[mask], labels2[mask]),
            "nmi": normalized_mutual_info_score(labels1[mask], labels2[mask]),
            "raw_agreement": (labels1[mask] == labels2[mask]).mean()
        }

    return agreement
```

#### 4.2.2 Consensus Threshold

Final taxonomy assignment requires:
- **Strong Consensus**: All 3 methods agree (weight = 1.0)
- **Majority Consensus**: 2/3 methods agree (weight = 0.8)
- **Weak Consensus**: Methods disagree (weight = 0.5, flag for sensitivity)

### 4.3 Cross-Validation for Compression Tolerance Analysis

#### 4.3.1 Stratified K-Fold Design

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_tolerance(
    tasks: list[dict],
    labels: np.ndarray,
    results: pd.DataFrame,
    n_splits: int = 5
) -> dict:
    """Cross-validate compression tolerance by category."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(tasks, labels)):
        # Compute tolerance metrics on test fold
        test_tasks = [tasks[i] for i in test_idx]
        test_results = results[results["task_id"].isin(
            [t["task_id"] for t in test_tasks]
        )]

        # Compute category-wise tolerance
        tolerance = compute_category_tolerance(test_results, labels[test_idx])
        tolerance["fold"] = fold
        fold_results.append(tolerance)

    # Aggregate across folds
    return aggregate_fold_results(fold_results)
```

---

## 5. Analysis Plan

### 5.1 Determining Ultra-Compressibility Threshold

#### 5.1.1 Definition

A task category C is **ultra-compressible** if and only if:

```
Pass@1(C, r=0.4) >= 0.30 * Pass@1(C, r=1.0)
```

Where:
- `Pass@1(C, r)` = mean pass rate for category C at compression ratio r
- `r=0.4` = ultra-compression threshold (60% token reduction)
- `0.30` = minimum quality retention factor (30% of baseline)

#### 5.1.2 Quality Retention Curve

For each category, fit a decay model:

```python
def fit_retention_curve(category_results: pd.DataFrame) -> dict:
    """Fit exponential decay model to quality vs compression."""
    from scipy.optimize import curve_fit

    # Model: Q(r) = Q_max * exp(-lambda * (1-r))
    def decay_model(r, Q_max, decay_lambda):
        return Q_max * np.exp(-decay_lambda * (1 - r))

    ratios = category_results["compression_ratio"].values
    pass_rates = category_results.groupby("compression_ratio")["passed"].mean()

    popt, pcov = curve_fit(
        decay_model,
        ratios,
        pass_rates,
        p0=[0.8, 2.0],
        bounds=([0, 0], [1, 10])
    )

    Q_max, decay_lambda = popt

    # Compute r_min: minimum ratio for 30% quality retention
    # Q(r_min) = 0.3 * Q_max
    # 0.3 * Q_max = Q_max * exp(-lambda * (1 - r_min))
    # 0.3 = exp(-lambda * (1 - r_min))
    # ln(0.3) = -lambda * (1 - r_min)
    # r_min = 1 + ln(0.3) / lambda

    r_min = 1 + np.log(0.3) / decay_lambda
    r_min = np.clip(r_min, 0.1, 1.0)

    return {
        "Q_max": Q_max,
        "decay_lambda": decay_lambda,
        "r_min_30pct": r_min,
        "is_ultra_compressible": r_min <= 0.4
    }
```

#### 5.1.3 Threshold Sensitivity Analysis

Test robustness of ultra-compressibility classification across:
- Quality thresholds: [0.20, 0.25, 0.30, 0.35, 0.40]
- Compression ratios: [0.35, 0.40, 0.45, 0.50]

Report stability matrix: % of categories that maintain classification across threshold variations.

### 5.2 Feature Importance Analysis

#### 5.2.1 Task-Level Features

| Feature | Description | Expected Correlation |
|---------|-------------|---------------------|
| `prompt_length` | Token count in prompt | Negative |
| `code_length` | Token count in solution | Negative |
| `cyclomatic_complexity` | McCabe complexity metric | Negative |
| `ast_depth` | Maximum depth of AST | Negative |
| `unique_token_ratio` | Unique tokens / Total tokens | Positive |
| `keyword_density` | Domain keywords / Total tokens | Positive |
| `has_signature` | Explicit function signature in prompt | Positive |
| `example_count` | Number of examples in prompt | Neutral |
| `nesting_depth` | Maximum loop/conditional nesting | Negative |

#### 5.2.2 Feature Extraction Pipeline

```python
import ast
import radon.complexity as cc

def extract_task_features(task: dict) -> dict:
    """Extract predictive features from MBPP task."""
    prompt = task["text"]
    code = task["code"]

    # Prompt features
    prompt_tokens = prompt.split()

    # Code complexity
    try:
        tree = ast.parse(code)
        complexity = cc.cc_visit(code)
        avg_cc = np.mean([c.complexity for c in complexity]) if complexity else 1
        ast_depth = compute_ast_depth(tree)
    except SyntaxError:
        avg_cc = np.nan
        ast_depth = np.nan

    return {
        "prompt_length": len(prompt_tokens),
        "code_length": len(code.split()),
        "cyclomatic_complexity": avg_cc,
        "ast_depth": ast_depth,
        "unique_token_ratio": len(set(prompt_tokens)) / len(prompt_tokens),
        "keyword_density": count_domain_keywords(prompt) / len(prompt_tokens),
        "has_signature": bool(re.search(r"def\s+\w+\s*\(", prompt)),
        "example_count": prompt.count(">>>") + prompt.count("Example"),
        "nesting_depth": compute_nesting_depth(code)
    }
```

---

## 6. Statistical Methods

### 6.1 Hypothesis Testing

#### 6.1.1 RQ1: Category Structure

**Test**: Silhouette Score Analysis
```python
from sklearn.metrics import silhouette_score

def test_cluster_validity(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Test if clusters represent meaningful structure."""

    # Overall silhouette score
    overall_silhouette = silhouette_score(embeddings, labels)

    # Per-cluster silhouette (check for poorly-defined clusters)
    sample_silhouettes = silhouette_samples(embeddings, labels)
    cluster_silhouettes = {
        label: sample_silhouettes[labels == label].mean()
        for label in np.unique(labels)
    }

    # Statistical test: compare to random baseline
    n_permutations = 1000
    random_silhouettes = []
    for _ in range(n_permutations):
        random_labels = np.random.permutation(labels)
        random_silhouettes.append(silhouette_score(embeddings, random_labels))

    p_value = (np.array(random_silhouettes) >= overall_silhouette).mean()

    return {
        "silhouette_score": overall_silhouette,
        "cluster_silhouettes": cluster_silhouettes,
        "vs_random_p": p_value,
        "is_significant": p_value < 0.01
    }
```

#### 6.1.2 RQ2: Compression Tolerance Differential

**Test 1**: Chi-Square Test of Independence
```python
from scipy.stats import chi2_contingency

def test_category_independence(results_df: pd.DataFrame) -> dict:
    """Test if pass/fail rates are independent of category at each ratio."""

    chi_square_results = {}

    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
        subset = results_df[results_df["compression_ratio"] == ratio]

        # Contingency table: category x pass/fail
        contingency = pd.crosstab(subset["category"], subset["passed"])

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Effect size: Cramer's V
        n = contingency.sum().sum()
        k = min(contingency.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))

        chi_square_results[ratio] = {
            "chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "cramers_v": cramers_v,
            "is_significant": p_value < 0.01
        }

    return chi_square_results
```

**Test 2**: ANOVA for Variance Decomposition
```python
from scipy.stats import f_oneway

def test_variance_decomposition(results_df: pd.DataFrame) -> dict:
    """Test if between-category variance exceeds within-category."""

    # At compression ratio r=0.4
    r04_data = results_df[results_df["compression_ratio"] == 0.4]

    # Group pass rates by category
    category_groups = [
        group["passed"].values
        for _, group in r04_data.groupby("category")
    ]

    # One-way ANOVA
    f_stat, p_value = f_oneway(*category_groups)

    # Effect size: eta-squared
    ss_between = sum(len(g) * (g.mean() - r04_data["passed"].mean())**2
                     for g in category_groups)
    ss_total = ((r04_data["passed"] - r04_data["passed"].mean())**2).sum()
    eta_squared = ss_between / ss_total

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "eta_squared": eta_squared,
        "is_significant": p_value < 0.01 and f_stat > 3.0
    }
```

#### 6.1.3 RQ3: Predictive Features

**Method 1**: Correlation Analysis
```python
from scipy.stats import pearsonr, spearmanr

def compute_feature_correlations(
    features_df: pd.DataFrame,
    tolerance_scores: np.ndarray
) -> pd.DataFrame:
    """Compute correlation between features and compression tolerance."""

    correlations = []
    for feature in features_df.columns:
        # Pearson (linear)
        r_pearson, p_pearson = pearsonr(features_df[feature], tolerance_scores)

        # Spearman (monotonic)
        r_spearman, p_spearman = spearmanr(features_df[feature], tolerance_scores)

        correlations.append({
            "feature": feature,
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "spearman_r": r_spearman,
            "spearman_p": p_spearman,
            "significant": (p_pearson < 0.01) or (p_spearman < 0.01)
        })

    return pd.DataFrame(correlations)
```

**Method 2**: Regression Analysis
```python
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

def build_predictive_model(
    features: np.ndarray,
    optimal_ratios: np.ndarray
) -> dict:
    """Build model to predict optimal compression ratio from features."""

    from sklearn.model_selection import cross_val_score

    # Model 1: Regularized Linear Regression
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge_scores = cross_val_score(ridge, features, optimal_ratios,
                                   cv=5, scoring='neg_mean_absolute_error')

    # Model 2: Random Forest (non-linear)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_scores = cross_val_score(rf, features, optimal_ratios,
                                cv=5, scoring='neg_mean_absolute_error')

    # Feature importance from RF
    rf.fit(features, optimal_ratios)

    return {
        "ridge_mae": -ridge_scores.mean(),
        "ridge_std": ridge_scores.std(),
        "rf_mae": -rf_scores.mean(),
        "rf_std": rf_scores.std(),
        "feature_importance": dict(zip(
            feature_names, rf.feature_importances_
        ))
    }
```

### 6.2 Multiple Testing Correction

Apply Benjamini-Hochberg FDR correction for:
- 7 categories x 6 compression ratios = 42 comparisons
- Target FDR: q = 0.05

```python
from statsmodels.stats.multitest import multipletests

def correct_multiple_tests(p_values: list[float], method='fdr_bh') -> np.ndarray:
    """Apply FDR correction to p-values."""
    rejected, corrected_p, _, _ = multipletests(p_values, method=method)
    return corrected_p
```

### 6.3 Power Analysis

#### 6.3.1 Sample Size Requirements

```python
from statsmodels.stats.power import TTestIndPower, GofChisquarePower

def compute_required_samples() -> dict:
    """Compute required sample sizes for adequate power."""

    # For t-test comparing two categories (effect size d=0.5, power=0.8)
    power_analysis = TTestIndPower()
    n_per_group = power_analysis.solve_power(
        effect_size=0.5,  # Medium effect
        alpha=0.01,
        power=0.80,
        ratio=1.0
    )

    # For chi-square (w=0.3, power=0.8, df=6)
    chi2_power = GofChisquarePower()
    n_chi2 = chi2_power.solve_power(
        effect_size=0.3,  # Medium effect
        alpha=0.01,
        power=0.80,
        n_bins=7  # 7 categories
    )

    return {
        "n_per_category_ttest": int(np.ceil(n_per_group)),  # ~105 per group
        "n_total_chi2": int(np.ceil(n_chi2)),  # ~245 total
        "mbpp_available": 974,  # Sufficient
        "trials_at_6_ratios": 974 * 6  # 5,844 trials
    }
```

---

## 7. Expected Findings

### 7.1 Primary Hypotheses

Based on pilot analysis and domain knowledge, we expect:

| Category | Expected r_min | Rationale |
|----------|---------------|-----------|
| T7: I/O & Formatting | 0.30-0.35 | Highly stereotypical patterns, minimal semantic content |
| T1: String Manipulation | 0.35-0.42 | Common operations, predictable structure |
| T2: List/Array Operations | 0.40-0.48 | Built-in functions, redundant descriptions |
| T5: Data Structure Manipulation | 0.45-0.52 | Type specifications important but compressible |
| T3: Arithmetic/Mathematical | 0.48-0.55 | Parameter names critical, less redundancy |
| T4: Control Flow Logic | 0.52-0.60 | Edge cases require explicit specification |
| T6: Algorithmic/Recursive | 0.58-0.70 | Algorithm name and structure critical |

### 7.2 Ultra-Compressible Subset

**Expected Result**: Categories T1, T2, T7 (and possibly T5) will tolerate r <= 0.4.

Estimated coverage: **35-45% of MBPP tasks**

### 7.3 Category-Specific Degradation Patterns

| Category | Degradation Mode at Ultra-Compression |
|----------|--------------------------------------|
| T1 | Incorrect edge case handling (empty string, unicode) |
| T2 | Wrong built-in function choice (sort vs sorted) |
| T3 | Missing base cases, off-by-one errors |
| T4 | Incomplete conditional branches |
| T5 | Type confusion (dict vs list of tuples) |
| T6 | Incorrect recursion base case, wrong algorithm |
| T7 | Format string errors, type coercion issues |

### 7.4 Feature Importance Predictions

Expected top-3 features for predicting compression tolerance:
1. **Cyclomatic complexity** (r = -0.45 to -0.55)
2. **Prompt unique token ratio** (r = +0.30 to +0.40)
3. **AST depth** (r = -0.35 to -0.45)

---

## 8. Practical Implications

### 8.1 Task-Aware Routing System

The validated taxonomy enables a production routing system:

```python
class TaskAwareRouter:
    """Route compression ratio based on detected task type."""

    def __init__(self, taxonomy_model: str = "trained_classifier.pkl"):
        self.classifier = load_classifier(taxonomy_model)
        self.ratio_map = {
            "T7_io": 0.35,
            "T1_string": 0.40,
            "T2_list": 0.45,
            "T5_data_struct": 0.50,
            "T3_math": 0.55,
            "T4_control": 0.55,
            "T6_algorithmic": 0.65,
            "T0_ambiguous": 0.60  # Conservative fallback
        }

    def route(self, prompt: str) -> float:
        """Determine optimal compression ratio for prompt."""
        category = self.classifier.predict(prompt)
        confidence = self.classifier.predict_proba(prompt).max()

        base_ratio = self.ratio_map[category]

        # Confidence adjustment: less confident -> more conservative
        if confidence < 0.7:
            base_ratio = min(base_ratio + 0.1, 0.8)

        return base_ratio
```

### 8.2 Expected Cost Savings

At enterprise scale (1B API calls/day):

| Scenario | Avg Ratio | Daily Tokens | Daily Cost | vs Baseline |
|----------|-----------|--------------|------------|-------------|
| Baseline (uniform r=0.6) | 0.60 | 300B | $3.0M | - |
| Task-aware routing | 0.48 | 240B | $2.4M | -20% |
| Ultra-compress eligible | 0.35 | 175B | $1.75M | -42% |

**Annual savings potential**: $200M-$450M at enterprise scale

### 8.3 Quality-Cost Tradeoff Visualization

```
Quality (Pass@1)
    ^
1.0 |    * * * * * * * * * * * *
    |   *                         T7_io
0.8 |  *     * * * * * * *
    | *    *                      T1_string
0.6 |*   *       * * * * *
    |   *      *                  T2_list
0.4 |  *     *        * * *
    | *    *       *              T6_algorithmic
0.2 |*   *      *
    |  *     *
0.0 +----------------------------> Compression Ratio (r)
    0.3  0.4  0.5  0.6  0.7  1.0

    [Ultra-Compressible Zone: r <= 0.4]
```

### 8.4 Integration with Plexor

The taxonomy directly informs Plexor's routing decisions:

```python
# In Plexor optimization pipeline
def optimize_prompt(prompt: str, mode: str = "balanced") -> str:
    # Step 1: Classify task type
    task_type = taxonomy_classifier.predict(prompt)

    # Step 2: Determine target compression based on mode and task
    if mode == "eco":
        target_ratio = ULTRA_COMPRESS_MAP.get(task_type, 0.5)
    elif mode == "balanced":
        target_ratio = BALANCED_MAP.get(task_type, 0.6)
    else:  # quality
        target_ratio = 0.8

    # Step 3: Apply LLMLingua-2 compression to target
    compressed = llmlingua.compress(prompt, target_ratio=target_ratio)

    return compressed
```

---

## 9. References

1. **Austin, J., et al. (2021)**. "Program Synthesis with Large Language Models." *arXiv preprint arXiv:2108.07732*. [MBPP benchmark paper]

2. **Jiang, Z., et al. (2023)**. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." *EMNLP 2023*.

3. **Pan, Y., et al. (2024)**. "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression." *ACL 2024*.

4. **McCabe, T. (1976)**. "A Complexity Measure." *IEEE Transactions on Software Engineering*.

5. **Halstead, M. (1977)**. "Elements of Software Science." *Elsevier*.

6. **Radon Documentation**. "Code Metrics in Python." https://radon.readthedocs.io/

7. **Cohen, J. (1960)**. "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement*.

8. **Fleiss, J. (1971)**. "Measuring Nominal Scale Agreement Among Many Raters." *Psychological Bulletin*.

9. **Chen, M., et al. (2021)**. "Evaluating Large Language Models Trained on Code." *arXiv preprint arXiv:2107.03374*. [HumanEval, related methodology]

10. **Feng, Z., et al. (2020)**. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." *EMNLP 2020*.

---

## 10. Appendices

### Appendix A: Complete Keyword Lists

[Full keyword dictionaries for each category - see `task_classification_analysis.py`]

### Appendix B: Annotation Guidelines (Full Document)

[Complete annotator training materials - to be created]

### Appendix C: Statistical Analysis Scripts

All analysis scripts available at:
`/research/paper-v3/experiments/task_classification_analysis.py`

### Appendix D: Raw Results Template

```json
{
  "experiment_id": "3B",
  "timestamp": "2026-01-XX",
  "dataset": "MBPP",
  "n_tasks": 974,
  "classification_results": {
    "method_a_heuristic": {...},
    "method_b_embedding": {...},
    "method_c_manual": {...},
    "consensus": {...}
  },
  "compression_tolerance": {
    "by_category": {...},
    "by_ratio": {...}
  },
  "statistical_tests": {
    "chi_square": {...},
    "anova": {...},
    "correlations": {...}
  },
  "model_performance": {
    "ridge_mae": ...,
    "rf_mae": ...
  }
}
```

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-18 | Initial design document | Dr. Michael Park |

---

*This study design document is prepared for NeurIPS 2026 submission in the Datasets and Benchmarks Track.*
