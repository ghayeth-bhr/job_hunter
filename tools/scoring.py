"""
scoring.py — deterministic opportunity scoring. No LLM, no network, no tokens.

Ported from pfzebi/PFE-Hunter (github.com/OussemaBenAmeur/pfzebi), a sibling
project solving the same problem (Tunisian ENISo student hunting PFE/research
internships) that reached the same conclusion this project's whole ranking
subsystem exists to work around: an LLM in the ranking loop measures noise,
not signal. Their own commit history documents it directly — with an LLM
removed and only four cheap terms left, every score across 328 real rows fell
between 0.32 and 0.43, with 125 rows tied at exactly 0.34. A score whose terms
don't vary is not a ranking.

The fix ported here is not "add an LLM back" -- it's "use the right terms,
with the right denominator". Skill fit measures what the POSTING asks for and
how much of that you have (not what fraction of your 30+ skills a three-line
posting happens to mention, which is ~10% for every posting and discriminates
nothing). Role fit strips stopwords and requires a DISCRIMINATING token
overlap, not just "engineer" matching "engineer". Seniority classification
uses the LEFTMOST marker in the title, not the highest-weight one, because
"Summer Intern, Director of Product" is an internship and a naive weighted
match would misclassify it as senior.

This module handles the INDUSTRY track (a normal job posting) and the PROGRAM
track (a funded program from config/programs.yaml, scored by deadline
urgency instead of posting freshness). It does not decide ELIGIBILITY
(country/visa/work-authorization) -- that stays a separate concern, handled
by main.py's deterministic pre-filter plus a narrow LLM pass reserved for
genuinely ambiguous cases. Folding eligibility into this score, as the
source project does via `evidence`, would blur two independent questions
("how good a fit" vs "can you actually apply") that this project already
treats as orthogonal via the INELIGIBLE tier.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone

# ════════════════════════════════════════════════════════════════════════════
#  SKILL VOCABULARY — curated, not token-frequency-derived
#
#  WHY A CURATED VOCABULARY. "What fraction of the candidate's skills appear
#  in this posting" is ~constant across every posting (no job asks for a
#  whole CV). The fix is to extract what the POSTING asks for, from a known
#  vocabulary, then ask how much of THAT the candidate has. Every entry below
#  is word-boundary anchored; short/ambiguous ones (Go, R, C) get a dedicated
#  context-sensitive check instead, since a bare-letter false positive claims
#  a skill match that isn't real (worse than missing a real one).
# ════════════════════════════════════════════════════════════════════════════

_SKILL_TOKENS = [
    # Languages
    r"Python", r"TypeScript", r"JavaScript", r"Java", r"Kotlin", r"Scala", r"Rust",
    r"C\+\+", r"C#", r"\.NET", r"MATLAB", r"Julia", r"Swift", r"PHP", r"Ruby", r"SQL",
    r"Bash", r"Shell scripting",
    # ML / DL frameworks
    r"PyTorch", r"Torch", r"TensorFlow", r"Keras", r"JAX", r"Flax",
    r"scikit-?learn", r"sklearn", r"XGBoost", r"LightGBM", r"CatBoost",
    r"Hugging ?Face", r"Transformers", r"Diffusers", r"timm", r"OpenCV", r"Detectron2",
    r"MMDetection", r"Ultralytics", r"YOLO(?:v\d+)?", r"StrongSORT", r"SAM ?2",
    r"Grounding ?DINO", r"SigLIP2?", r"RT-?DETR", r"R-?CNN", r"ResNet", r"ViT",
    r"Stable Diffusion", r"GANs?", r"VAEs?",
    # LLM / GenAI
    r"LLMs?", r"large language models?", r"RAG", r"retrieval[- ]augmented generation",
    r"LangChain", r"LangGraph", r"LlamaIndex", r"Haystack", r"DSPy",
    r"vLLM", r"Ollama", r"Mistral", r"Llama", r"GPT-?4o?", r"Claude", r"Gemini",
    r"prompt engineering", r"fine-?tuning", r"LoRA", r"PEFT", r"RLHF", r"quantization",
    r"embeddings?", r"vector (?:database|store|search)", r"Pinecone", r"Weaviate",
    r"Qdrant", r"ChromaDB", r"FAISS", r"Milvus", r"multi-?agent", r"agentic", r"MCP",
    r"function calling", r"n8n", r"vision[- ]language models?",
    # CV / NLP / classical ML
    r"computer vision", r"object detection", r"image segmentation", r"OCR",
    r"PaddleOCR", r"LayoutLM", r"Qwen-?VL", r"multimodal", r"NLP",
    r"natural language processing", r"named entity recognition",
    r"speech recognition", r"ASR", r"time ?series", r"forecasting",
    r"anomaly detection", r"fraud detection", r"recommender systems?",
    r"reinforcement learning", r"graph neural networks?", r"GNNs?",
    r"clustering", r"DBSCAN", r"Isolation Forest", r"Bayesian",
    # Data
    r"pandas", r"NumPy", r"SciPy", r"Polars", r"Matplotlib", r"Seaborn", r"Plotly",
    r"Statsmodels", r"Spark", r"PySpark", r"Hadoop", r"Kafka", r"Redpanda", r"Flink",
    r"Airflow", r"dbt", r"Databricks", r"Snowflake", r"BigQuery", r"Redshift",
    r"PostgreSQL", r"Postgres", r"MySQL", r"MongoDB", r"Redis", r"Elasticsearch",
    r"DuckDB", r"Neo4j",
    # MLOps / infra
    r"MLOps", r"MLflow", r"Weights ?& ?Biases", r"W&B", r"DVC", r"Kubeflow", r"SageMaker",
    r"Vertex AI", r"ONNX", r"TensorRT", r"Triton", r"CUDA", r"Ray", r"Dask",
    r"Docker", r"Kubernetes", r"k8s", r"Terraform", r"Ansible", r"Helm",
    r"AWS", r"GCP", r"Google Cloud", r"Azure", r"CI/CD", r"GitHub Actions", r"GitLab CI",
    r"Jenkins", r"Prometheus", r"Grafana", r"Linux", r"Git", r"NVIDIA Jetson",
    r"DeepStream", r"MQTT", r"OAuth2",
    # Backend / product surface
    r"FastAPI", r"Flask", r"Django", r"Spring Boot", r"Node\.?js", r"Express",
    r"React", r"Next\.?js", r"Vue\.?js", r"Angular", r"Svelte", r"Flutter",
    r"REST APIs?", r"GraphQL", r"gRPC", r"microservices", r"event-driven architecture",
    r"Saga pattern", r"transactional outbox", r"distributed systems",
    # Research-flavoured
    r"LaTeX", r"publications?", r"peer[- ]review", r"research experience",
]

_SKILL_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(_SKILL_TOKENS) + r")(?!\w)", re.IGNORECASE
)

# Every spelling that means the same skill collapses to one display name, so
# "k8s" and "Kubernetes" in the same posting don't count twice, and a profile
# saying "PyTorch" isn't missed by a JD saying "torch".
_CANONICAL: dict[str, str] = {
    "torch": "PyTorch", "pytorch": "PyTorch",
    "sklearn": "scikit-learn", "scikit-learn": "scikit-learn", "scikitlearn": "scikit-learn",
    "hugging face": "Hugging Face", "huggingface": "Hugging Face",
    "k8s": "Kubernetes", "kubernetes": "Kubernetes",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
    "node.js": "Node.js", "nodejs": "Node.js",
    "next.js": "Next.js", "nextjs": "Next.js",
    "vue.js": "Vue.js", "vuejs": "Vue.js",
    "llm": "LLMs", "llms": "LLMs", "large language model": "LLMs", "large language models": "LLMs",
    "rag": "RAG", "retrieval-augmented generation": "RAG", "retrieval augmented generation": "RAG",
    "fine tuning": "Fine-tuning", "finetuning": "Fine-tuning", "fine-tuning": "Fine-tuning",
    "nlp": "NLP", "natural language processing": "NLP",
    "computer vision": "Computer Vision", "object detection": "Object Detection",
    "anomaly detection": "Anomaly Detection", "fraud detection": "Fraud Detection",
    "graph neural network": "GNNs", "graph neural networks": "GNNs", "gnn": "GNNs", "gnns": "GNNs",
    "reinforcement learning": "Reinforcement Learning",
    "multi-agent": "Multi-Agent", "multiagent": "Multi-Agent", "agentic": "Multi-Agent",
    "rt-detr": "RT-DETR", "rtdetr": "RT-DETR",
    "r-cnn": "R-CNN", "rcnn": "R-CNN",
    "qwen-vl": "Qwen-VL", "qwenvl": "Qwen-VL",
    "weights & biases": "Weights & Biases", "weights and biases": "Weights & Biases", "w&b": "Weights & Biases",
    "google cloud": "GCP", "gcp": "GCP", "aws": "AWS", "azure": "Azure",
    "ci/cd": "CI/CD", "github actions": "GitHub Actions", "gitlab ci": "GitLab CI",
    "rest api": "REST APIs", "rest apis": "REST APIs",
    "time series": "Time Series", "timeseries": "Time Series",
    "spring boot": "Spring Boot", "prompt engineering": "Prompt Engineering",
    "vector database": "Vector Search", "vector store": "Vector Search", "vector search": "Vector Search",
    "gan": "GANs", "gans": "GANs", "vae": "VAEs", "vaes": "VAEs",
    "c++": "C++", "c#": "C#", ".net": ".NET",
    "recommender system": "Recommender Systems", "recommender systems": "Recommender Systems",
    "publication": "Publications", "publications": "Publications",
    "peer-review": "Peer Review", "peer review": "Peer Review",
    "shell scripting": "Bash", "bash": "Bash",
    "image segmentation": "Segmentation", "named entity recognition": "NER",
    "speech recognition": "ASR", "asr": "ASR",
    "yolo": "YOLO", "strongsort": "StrongSORT", "sam 2": "SAM 2", "sam2": "SAM 2",
    "grounding dino": "Grounding DINO", "siglip2": "SigLIP2", "siglip": "SigLIP2",
    "vision-language models": "Vision-Language Models", "nvidia jetson": "NVIDIA Jetson",
    "deepstream": "DeepStream", "mqtt": "MQTT", "oauth2": "OAuth2",
    "distributed systems": "Distributed Systems", "microservices": "Microservices",
    "event-driven architecture": "Event-Driven Architecture", "saga pattern": "Saga Pattern",
    "transactional outbox": "Transactional Outbox", "redpanda": "Redpanda",
}

# Ambiguous short names that a bare word-boundary match over-fires on. Each
# needs a context clue -- "Go" appears in nearly every posting as a plain
# word, and "R" is a letter. Missing one costs a little signal; inventing one
# is a false claim, which is the worse failure mode.
_AMBIGUOUS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<![\w-])Go(?:lang)?\s*(?:\(|,|/|·|and\b|programming\b|developer\b|experience\b)", re.IGNORECASE), "Go"),
    (re.compile(r"(?<![\w-])Golang(?![\w-])", re.IGNORECASE), "Go"),
    (re.compile(r"(?<![\w-])R\s*(?:/|,)\s*(?:Python|SAS|Stata)", re.IGNORECASE), "R"),
    (re.compile(r"(?<![\w-])(?:in|with|using)\s+R(?![\w-])", re.IGNORECASE), "R"),
    (re.compile(r"(?<![\w-])C\s*(?:/|,|\sor\s)\s*C\+\+", re.IGNORECASE), "C"),
]


# Lowercased spelling -> the vocabulary's own literal casing, derived from
# _SKILL_TOKENS itself (stripping regex escapes/optionality markers) so the
# token list above is the single place a name's casing is decided. Without
# this, the naive fallback below mis-cases mixed-capital terms -- caught
# live: "OpenCV" (correctly spelled in _SKILL_TOKENS) canonicalized to
# "Opencv" via a bare str.title() call, which would silently break matching
# between a candidate's declared "OpenCV" and a posting's extracted "OpenCV"
# the instant either one took a different path through this function.
def _strip_regex_syntax(token: str) -> str:
    return re.sub(r"\\|\?:|\?|\(|\)", "", token)


_DISPLAY: dict[str, str] = {
    _strip_regex_syntax(t).lower(): _strip_regex_syntax(t) for t in _SKILL_TOKENS
}

# Hand-set casing for names the regex source can't spell literally (it's
# either optional/alternated, or the source uses different casing than the
# desired display form).
_DISPLAY_OVERRIDES: dict[str, str] = {
    "sql": "SQL", "nlp": "NLP", "ocr": "OCR", "asr": "ASR", "api": "API",
    "n8n": "n8n", "javascript": "JavaScript", "typescript": "TypeScript",
    "langchain": "LangChain", "langgraph": "LangGraph", "llamaindex": "LlamaIndex",
    "mlops": "MLOps", "mlflow": "MLflow", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "numpy": "NumPy", "scipy": "SciPy", "fastapi": "FastAPI", "pyspark": "PySpark",
    "duckdb": "DuckDB", "neo4j": "Neo4j", "chromadb": "ChromaDB", "faiss": "FAISS",
    "paddleocr": "PaddleOCR", "layoutlm": "LayoutLM", "tensorrt": "TensorRT",
    "onnx": "ONNX", "cuda": "CUDA", "dbt": "dbt", "jax": "JAX", "timm": "timm",
    "vllm": "vLLM", "lora": "LoRA", "peft": "PEFT", "rlhf": "RLHF", "mcp": "MCP",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "opencv": "OpenCV", "detectron2": "Detectron2", "mmdetection": "MMDetection",
    "ultralytics": "Ultralytics", "resnet": "ResNet", "vit": "ViT", "gpt": "GPT",
    "latex": "LaTeX", "linux": "Linux", "git": "Git", "matlab": "MATLAB", "php": "PHP",
    "graphql": "GraphQL", "kubeflow": "Kubeflow", "sagemaker": "SageMaker",
    "databricks": "Databricks", "bigquery": "BigQuery", "postgresql": "PostgreSQL",
    "mysql": "MySQL", "mongodb": "MongoDB", "elasticsearch": "Elasticsearch",
    "ner": "NER", "gnns": "GNNs", "vaes": "VAEs", "gans": "GANs", "yolo": "YOLO",
    "strongsort": "StrongSORT", "siglip2": "SigLIP2", "siglip": "SigLIP2",
    "nvidia jetson": "NVIDIA Jetson", "deepstream": "DeepStream", "mqtt": "MQTT",
    "oauth2": "OAuth2", "redpanda": "Redpanda",
}


# Version-suffixed mentions ("YOLOv8", "YOLO11") must canonicalize the same
# as the bare name -- a posting asking for "YOLO" and a candidate who has
# "YOLOv8" experience are the same skill, and matching should not depend on
# which version either of them happened to write. Caught live: without this,
# "YOLOv8" and "YOLO11" canonicalized to themselves (lowercase, unmatched)
# instead of the single "YOLO" that bare mentions and _CANONICAL both use.
_VERSION_SUFFIX = re.compile(r"^([a-z]+)v\d+$")   # "yolov8" -> "yolo"
_BARE_NUMBER_SUFFIX = re.compile(r"^([a-z]+)\d+$")  # "yolo11" -> "yolo"


def _canonicalize(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    if key in _CANONICAL:
        return _CANONICAL[key]
    if key in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[key]
    if key in _DISPLAY:
        return _DISPLAY[key]
    for pattern in (_VERSION_SUFFIX, _BARE_NUMBER_SUFFIX):
        vm = pattern.match(key)
        # Only strip the suffix when the base name is actually a KNOWN skill
        # -- otherwise this would silently mangle any unrelated word that
        # happens to end in digits.
        if vm and vm.group(1) in _DISPLAY_OVERRIDES:
            return _DISPLAY_OVERRIDES[vm.group(1)]
        if vm and vm.group(1) in _DISPLAY:
            return _DISPLAY[vm.group(1)]
    return key.title() if " " in key or key.isalpha() else key.upper() if len(key) <= 4 else key


def extract_skills(text: str | None) -> set[str]:
    """Every canonical skill named anywhere in `text`."""
    s = text or ""
    if not s:
        return set()
    found = {_canonicalize(m.group(0)) for m in _SKILL_PATTERN.finditer(s)}
    for pattern, name in _AMBIGUOUS:
        if pattern.search(s):
            found.add(name)
    return found


def profile_skill_set(declared: list[str], free_text: str = "") -> set[str]:
    """The candidate's skills, canonicalized -- from both the declared list
    and free text (project descriptions), since a declared list names
    "PyTorch" once but "YOLO"/"DBSCAN" only ever show up in project text."""
    out: set[str] = set()
    for raw in declared:
        hits = extract_skills(raw)
        if hits:
            out |= hits
        elif raw.strip():
            out.add(_canonicalize(raw))
    out |= extract_skills(free_text)
    return out


def skill_fit(jd_text: str | None, profile: set[str], org: str = "") -> dict:
    """0..1 -- of what the POSTING asks for, how much the candidate has.

    The denominator is the posting, not the CV -- "you have 6 of the 8 things
    they asked for" discriminates between jobs; "this job mentions 6 of your
    34 skills" is roughly constant across all of them.

    A posting naming nothing recognizable returns a neutral 0.5, not 0 --
    many real listings are three sentences long, and scoring those as a total
    mismatch would rank a terse posting at a great company below a
    keyword-stuffed repost.
    """
    asked = extract_skills(jd_text)

    # The employer's own name is not a requirement -- a Mistral posting says
    # "Mistral" because that's who they are, not because they're asking for
    # it, and several vocabulary entries double as company names (Mistral,
    # Llama, Claude, Gemini, Databricks, Snowflake).
    if org:
        org_words = {w for w in re.split(r"[^a-z0-9+#.]+", org.lower()) if w}
        asked = {s for s in asked if s.lower() not in org_words}

    if not asked:
        return {"score": 0.5, "matched": [], "missing": [], "unknown": True}

    matched = sorted(s for s in asked if s in profile)
    missing = sorted(s for s in asked if s not in profile)

    # Bayesian-smoothed, not a raw ratio -- a posting naming exactly one skill
    # you happen to have is not a 100% match. Adding pseudo-observations at
    # the neutral prior makes a small denominator behave like the weak
    # evidence it is: 1/1 -> 0.67, 6/8 -> 0.70, 12/14 -> 0.81. A long,
    # well-matched list still wins, which is the point.
    PRIOR_WEIGHT, PRIOR = 2, 0.5
    score = (len(matched) + PRIOR_WEIGHT * PRIOR) / (len(asked) + PRIOR_WEIGHT)
    return {"score": score, "matched": matched, "missing": missing, "unknown": False}


# ════════════════════════════════════════════════════════════════════════════
#  SENIORITY TIER — leftmost marker decides, not highest weight
#
#  "Summer Intern, Director of Product" is an internship. English titles put
#  the level word first; ranking by weight lets a stray "Director" (naming
#  the team lead the intern sits beside) misclassify the whole posting as
#  senior. Two guards below cover the exceptions this rule itself creates.
# ════════════════════════════════════════════════════════════════════════════

_TIER_MATCHERS: list[tuple[re.Pattern, str, int]] = [
    (re.compile(r"\bchief\b", re.I), "senior", 4),
    (re.compile(r"\bvp\b", re.I), "senior", 4),
    (re.compile(r"\bvice\s+president\b", re.I), "senior", 4),
    (re.compile(r"\bdirector\b", re.I), "senior", 4),
    (re.compile(r"\bprincipal\b", re.I), "senior", 4),
    (re.compile(r"\bstaff\b", re.I), "senior", 4),
    (re.compile(r"\blead\b", re.I), "senior", 4),
    (re.compile(r"\bsenior\b", re.I), "senior", 4),
    (re.compile(r"\bsr\.?\b", re.I), "senior", 4),
    (re.compile(r"\bhead\s+of\b", re.I), "senior", 4),
    (re.compile(r"\b[a-z]{2,}[\s-](iii|iv|v)\b", re.I), "senior", 4),
    (re.compile(r"\bmid-level\b", re.I), "mid", 3),
    (re.compile(r"\bmid\b", re.I), "mid", 3),
    (re.compile(r"\b[a-z]{2,}[\s-]ii\b", re.I), "mid", 3),
    (re.compile(r"\b(l4|l5)\b", re.I), "mid", 3),
    (re.compile(r"\bentry-level\b", re.I), "entry", 2),
    (re.compile(r"\bentry\b", re.I), "entry", 2),
    (re.compile(r"\bassociate\b", re.I), "entry", 2),
    (re.compile(r"\bjunior\b", re.I), "entry", 2),
    (re.compile(r"\b[a-z]{2,}[\s-]i\b", re.I), "entry", 2),
    (re.compile(r"\b(l1|l2)\b", re.I), "entry", 2),
    (re.compile(r"\binternship\b", re.I), "intern", 1),
    (re.compile(r"\bintern\b", re.I), "intern", 1),
    (re.compile(r"\btrainee\b", re.I), "intern", 1),
    (re.compile(r"\bco-op\b", re.I), "intern", 1),
    (re.compile(r"\bpraktik", re.I), "intern", 1),       # Praktikum/Praktikant
    (re.compile(r"\bwerkstudent", re.I), "intern", 1),
    (re.compile(r"\bstagi?air", re.I), "intern", 1),      # stagiaire/stagiair
    (re.compile(r"\bstage\b", re.I), "intern", 1),
    (re.compile(r"\balternance\b", re.I), "intern", 1),
    (re.compile(r"\bapprenti", re.I), "intern", 1),
    (re.compile(r"\bbecari", re.I), "intern", 1),         # becario/becaria
    (re.compile(r"\bpr[aá]ctica", re.I), "intern", 1),
    (re.compile(r"\btirocinio\b", re.I), "intern", 1),
    (re.compile(r"\bresidency\b", re.I), "intern", 1),
    (re.compile(r"\bfellowship\b", re.I), "intern", 1),
]
_GRADUATE_PROGRAM = re.compile(r"\bgraduate\b.*\b(program|scheme|cohort)\b", re.I)
_ASSOCIATE_SENIOR_AFTER = re.compile(r"\b(director|vice\s+president|vp|principal|partner|chief|head\s+of)\b", re.I)
_INTERN_BRIDGE = re.compile(
    r"\b(?:intern(?:ship)?|trainee|co-op|graduate|junior|entry(?:-level)?)\s+"
    r"(?:program|programme|scheme|talent|cohort)\b", re.I,
)
_SENIOR_AFTER_BRIDGE = re.compile(r"\b(chief|vp|vice\s+president|director|principal|staff|lead|senior|sr\.?|head\s+of|partner)\b", re.I)


def classify_tier(title: str | None) -> str:
    """One of 'intern'/'entry'/'mid'/'senior'. 'mid' is the unmatched bucket,
    not a claim the role is mid-level."""
    if not isinstance(title, str) or not title:
        return "mid"

    clean = re.sub(r"\bA\.I\.?\b", "AI", title, flags=re.I)
    clean = re.sub(r"\bI\.T\.?\b", "IT", clean, flags=re.I)

    # Guard: "Associate [X] Director/VP/Principal" is senior -- associate
    # qualifies a senior band, it doesn't demote it. Checked before the
    # leftmost-marker loop, since "associate" would otherwise win at index 0.
    m = re.search(r"\bassociate\b", clean, re.I)
    if m and _ASSOCIATE_SENIOR_AFTER.search(clean[m.end():]):
        return "senior"

    # Guard: [intern marker] + [program bridge noun] + [senior noun] is
    # senior -- an "Intern Program Director" manages an internship.
    if _INTERN_BRIDGE.search(clean):
        after = _INTERN_BRIDGE.sub(" ", clean)
        if _SENIOR_AFTER_BRIDGE.search(after):
            return "senior"

    best_tier, best_index, best_weight = None, None, -1
    for pattern, tier, weight in _TIER_MATCHERS:
        match = pattern.search(clean)
        if not match:
            continue
        index = match.start()
        if best_index is None or index < best_index or (index == best_index and weight > best_weight):
            best_tier, best_index, best_weight = tier, index, weight

    # "Graduate" alone is ambiguous ("Graduate Engineer" is a real hire); it
    # only counts as intern-tier with a program qualifier.
    gm = _GRADUATE_PROGRAM.search(clean)
    if gm:
        idx = re.search(r"\bgraduate\b", clean, re.I).start()
        if best_index is None or idx < best_index:
            best_tier, best_index = "intern", idx

    return best_tier or "mid"


def required_years(text: str | None) -> int | None:
    """Years of experience demanded, or None. '3+ years production ML' is not
    an internship however it's titled; 'familiarity with' is reachable."""
    m = re.search(
        r"(\d+)\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?|ans|jahre)\b[^.]{0,40}?"
        r"(?:experience|exp\b|erfahrung|exp[ée]rience)",
        text or "", re.I,
    )
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def level_fit(title: str, description: str) -> dict:
    """Is this actually reachable for a final-year student? The seniority
    pre-filter already dropped explicit seniors -- this grades what survived."""
    tier = classify_tier(title)
    years = required_years(description)
    if years is not None and years >= 3:
        return {"score": 0.05, "note": f"asks for {years}+ years of experience"}
    if years is not None and years >= 2:
        return {"score": 0.3, "note": f"asks for {years}+ years of experience"}
    if tier == "intern":
        return {"score": 1.0, "note": "explicitly an internship"}
    if tier == "entry":
        return {"score": 0.8, "note": "entry-level"}
    if tier == "senior":
        return {"score": 0.05, "note": "reads as senior-level"}
    return {"score": 0.5, "note": "level not stated"}


# ════════════════════════════════════════════════════════════════════════════
#  ROLE FIT — stopword-stripped, seniority-gated, discriminating overlap
# ════════════════════════════════════════════════════════════════════════════

_SENIORITY_TOKENS = {
    "junior", "mid", "middle", "senior", "staff", "principal", "lead", "head",
    "chief", "associate", "intern", "internship", "entry", "graduate", "trainee",
    "apprentice", "stage", "stagiaire", "praktikum", "praktikant", "werkstudent",
    "alternance", "alternant", "thesis", "pfe",
}
_STUDENT_LEVEL = {
    "intern", "internship", "entry", "graduate", "trainee", "apprentice",
    "stage", "stagiaire", "praktikum", "praktikant", "werkstudent",
    "alternance", "alternant", "thesis", "pfe", "junior", "associate",
}
_ROLE_STOPWORDS = _SENIORITY_TOKENS | {
    "level", "remote", "hybrid", "onsite", "contract", "contractor", "freelance",
    "fulltime", "parttime", "permanent", "temporary", "full", "part", "time",
    "role", "position", "opportunity", "team", "based", "months", "month",
    "repost", "reposted", "relisted", "new", "urgent",
    "london", "berlin", "paris", "madrid", "barcelona", "amsterdam", "dublin",
    "munich", "munchen", "zurich", "geneva", "lausanne", "brussels",
    "toronto", "montreal", "vancouver", "dubai", "riyadh", "doha", "singapore",
    "europe", "emea", "apac", "france", "germany", "canada", "switzerland",
    "netherlands", "belgium", "spain", "italy", "poland", "sweden", "denmark",
    "with", "from", "into", "over", "this", "that", "and", "for", "the",
}
_BASELINE_TOKENS = {
    "software", "engineer", "engineering", "developer", "development", "manager",
    "architect", "analyst", "designer", "consultant", "specialist", "scientist",
    "platform", "systems", "system", "services", "service", "technology",
    "backend", "frontend", "fullstack", "stack", "product", "technical",
}
_SHORT_SPECIALTY = {"ai", "ml", "nlp", "cv", "llm", "api", "gpu", "sre", "ux", "ui", "ds"}
_GENDER_TAG = re.compile(r"\b[mfdhwx](?:\s*[/·]\s*[mfdhwx]){1,3}\b", re.I)
_SLASH_PAIR = re.compile(r"\b([a-z0-9]{1,3})/([a-z0-9]{1,3})\b", re.I)
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_title(value: str | None) -> str:
    s = (value or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # strip accents
    return unicodedata.normalize("NFC", s)


def _role_tokens(role: str | None) -> list[str]:
    s = _normalize_title(role)
    s = _GENDER_TAG.sub(" ", s)
    s = _SLASH_PAIR.sub(r"\1 \2", s)
    s = _NON_WORD.sub(" ", s)
    return [w for w in s.split() if (len(w) > 3 or w in _SHORT_SPECIALTY) and w not in _ROLE_STOPWORDS]


def _seniority_in(title: str | None) -> set[str]:
    s = _NON_WORD.sub(" ", _normalize_title(title))
    return {w for w in s.split() if w in _SENIORITY_TOKENS}


def _seniority_compatible(a: str | None, b: str | None) -> bool:
    sa, sb = _seniority_in(a), _seniority_in(b)
    if sa and sb:
        return bool(sa & _STUDENT_LEVEL) == bool(sb & _STUDENT_LEVEL)
    return True  # only one side states a level -- allow it


def role_similarity(title: str | None, target: str | None) -> float:
    if not _seniority_compatible(title, target):
        return 0.0
    a, b = list(dict.fromkeys(_role_tokens(title))), list(dict.fromkeys(_role_tokens(target)))
    if not a or not b:
        return 0.0
    set_b = set(b)
    overlap = [w for w in a if w in set_b]
    if not overlap:
        return 0.0
    discriminating = [w for w in overlap if w not in _BASELINE_TOKENS]
    if not discriminating:
        return 0.0
    union = len(set(a) | set(b))
    jaccard = len(overlap) / union
    specificity = len(discriminating) / len(overlap)
    return min(1.0, jaccard * (0.6 + 0.4 * specificity) * 1.6)


def role_fit(title: str | None, target_roles: list[str]) -> dict:
    """The closest of the candidate's target roles, and how close it is."""
    best, score = None, 0.0
    for target in target_roles:
        s = role_similarity(title, target)
        if s > score:
            score, best = s, target
    return {"score": score, "best": best}


# ════════════════════════════════════════════════════════════════════════════
#  LEGITIMACY + INTERNATIONAL-FRIENDLINESS SIGNALS
# ════════════════════════════════════════════════════════════════════════════

_SCAM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btraining fee\b|\bregistration fee\b|\bsecurity deposit\b|\bpay (?:a|the) fee\b", re.I), "asks for a fee"),
    (re.compile(r"\bunpaid\b(?![^.]{0,40}\b(?:academic|credit)\b)", re.I), "explicitly unpaid"),
    (re.compile(r"\bequity[- ]only\b|\bin lieu of (?:salary|pay)\b", re.I), "equity instead of pay"),
    (re.compile(r"@(?:gmail|yahoo|hotmail|outlook|proton(?:mail)?)\.com", re.I), "personal email domain"),
    (re.compile(r"\bwhatsapp\b[^.]{0,30}\b(?:apply|contact|interview)\b", re.I), "hiring over WhatsApp"),
]


def legitimacy_flags(text: str | None) -> list[str]:
    """Red flags shared by unpaid/scam 'internship' postings. Applied as a
    MULTIPLICATIVE penalty on the final score, never averaged in -- a scam
    posting that happens to match every skill must not rank first."""
    s = text or ""
    return [label for pattern, label in _SCAM_PATTERNS if pattern.search(s)]


_INTL_POSITIVE = [
    re.compile(r"\bvisa sponsorship\b", re.I), re.compile(r"\bwe sponsor\b", re.I),
    re.compile(r"\bsponsorship (?:is )?available\b", re.I),
    re.compile(r"\brelocation (?:package|support|assistance)\b", re.I),
    re.compile(r"\bwork permit\b[^.]{0,40}\bsupport\b", re.I),
    re.compile(r"\binternational (?:students?|applicants?|candidates?) (?:are )?welcome\b", re.I),
    re.compile(r"\bconvention de stage\b", re.I), re.compile(r"\berasmus\b", re.I),
]
_INTL_NEGATIVE = [
    re.compile(r"\bno (?:visa )?sponsorship\b", re.I),
    re.compile(r"\bmust (?:already )?(?:have|hold)[^.]{0,40}\b(?:work authorization|right to work)\b", re.I),
    re.compile(r"\bcitizens? only\b", re.I), re.compile(r"\bsecurity clearance\b", re.I),
]


def intl_evidence(text: str | None) -> dict:
    """Sponsorship/international-eligibility language found in the text.
    Used two ways in this project: as a soft SCORE signal (see score_opportunity),
    and as a strong EXCEPTION signal for the eligibility check in main.py --
    e.g. 'convention de stage' or 'we sponsor' should skip the eligibility LLM
    call entirely rather than spend a call reasoning about something the text
    already states outright."""
    s = text or ""
    signals: list[str] = []
    score = 0.5  # unknown is neutral, never a penalty
    for pattern in _INTL_POSITIVE:
        m = pattern.search(s)
        if m:
            signals.append(m.group(0))
            score = 1.0
    for pattern in _INTL_NEGATIVE:
        m = pattern.search(s)
        if m:
            signals.append(f"(negative) {m.group(0)}")
            score = min(score, 0.15)
    return {"score": score, "signals": signals}


# ════════════════════════════════════════════════════════════════════════════
#  RECENCY, CONFIDENCE, AND THE WEIGHTED COMBINER
# ════════════════════════════════════════════════════════════════════════════


def recency_score(posted_at: str | None, max_age_days: int = 90, now: datetime | None = None) -> float | None:
    """1.0 today, decaying to 0 at max_age_days. No date on file is unknown,
    not a penalty -- returns None so `combine` redistributes its weight."""
    if not posted_at:
        return None
    try:
        t = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    now = now or datetime.now(timezone.utc)
    age_days = (now - t).total_seconds() / 86_400
    if age_days <= 0:
        return 1.0
    horizon = max_age_days if max_age_days > 0 else 90
    return max(0.0, 1 - age_days / horizon)


def description_confidence(description: str | None, track: str = "industry") -> float:
    """How much we actually know about this row, 0..1, from how much body
    text the source gave us. EXISTS BECAUSE A NAIVE SCORER REWARDS IGNORANCE:
    measured on real data, postings with NO description outscored postings
    with a full one, because a title-only row's skill term gets redistributed
    onto title/level, both of which max out trivially for "AI Intern". A
    program row is exempt -- its value is a deadline, not a job description."""
    if track == "program":
        return 1.0
    n = len((description or "").strip())
    if n == 0:
        return 0.45
    if n < 200:
        return 0.6
    if n < 800:
        return 0.85
    return 1.0


def shrink_to_neutral(raw: float, confidence: float) -> float:
    """Pull a score toward 0.5 in proportion to how little we know. Shrinkage,
    not a multiplier -- weak evidence should make the score less extreme in
    BOTH directions, not just penalize it downward."""
    return 0.5 + (raw - 0.5) * confidence


def combine(terms: list[tuple[float | None, float]]) -> float:
    """Weighted mean over KNOWN terms only. A None term's weight is dropped
    and redistributed, rather than substituted with 0.5 -- substituting the
    midpoint pulls every partially-known row toward the center, which is
    exactly how a naive scorer collapses into a narrow, undiscriminating band."""
    total, weight = 0.0, 0.0
    for value, w in terms:
        if value is None:
            continue
        total += value * w
        weight += w
    return total / weight if weight > 0 else 0.5


# ════════════════════════════════════════════════════════════════════════════
#  THE SCORER
# ════════════════════════════════════════════════════════════════════════════

_INDUSTRY_WEIGHTS = {"skill": 0.40, "role": 0.25, "level": 0.20, "recency": 0.15}
_PROGRAM_WEIGHTS = {"deadline": 0.60, "topic": 0.40}

# Tier thresholds kept aligned with this project's existing 1-10 vocabulary
# (TOP PICKS 8-10, GOOD FITS 5-7, WORTH EXPLORING 3-4, SKIP 0-2) so nothing
# downstream (report rendering, Telegram formatting, existing tests) needs to
# change for what's now a 0..1 float scaled to a 0-10 display score.
def _tier_of(score_0_to_10: float) -> str:
    if score_0_to_10 >= 8:
        return "TOP PICKS"
    if score_0_to_10 >= 5:
        return "GOOD FITS"
    if score_0_to_10 >= 3:
        return "WORTH EXPLORING"
    return "SKIP"


def program_deadline_score(deadline_iso: str | None, now: datetime | None = None) -> dict:
    """A funded program's value is a deadline you can still make."""
    if not deadline_iso:
        return {"score": 0.5, "note": "no deadline on file"}
    now = now or datetime.now(timezone.utc)
    try:
        d = datetime.fromisoformat(deadline_iso).replace(tzinfo=timezone.utc)
    except ValueError:
        return {"score": 0.5, "note": "unparseable deadline"}
    days = (d - now).days
    if days < 0:
        return {"score": 0.0, "note": "deadline passed"}
    if days <= 21:
        return {"score": 1.0, "note": f"closes in {days} days"}
    if days <= 60:
        return {"score": 0.85, "note": f"closes in {days} days"}
    return {"score": 0.6, "note": f"closes in {days} days"}


def score_opportunity(item: dict, profile_skills: set[str], target_roles: list[str]) -> dict:
    """Score one opportunity dict (main.py's `condensed` entry shape).

    Returns {score (0-10, matches this project's existing 1-10 vocabulary),
    tier, match_reasons, concerns, matched_skills, missing_skills, flags}.
    Deterministic: identical input always produces identical output, so
    there is no double-scoring, no DISPUTED tier, no retry -- those existed
    specifically to cope with an LLM's run-to-run inconsistency, which a pure
    function does not have.
    """
    title = item.get("title", "") or ""
    description = item.get("raw_content") or item.get("snippet") or ""
    body = f"{title}\n{description}"
    org = item.get("company", "") or ""
    track = item.get("track", "industry")

    flags = legitimacy_flags(body)
    reasons: list[str] = []
    matched_skills: list[str] = []
    missing_skills: list[str] = []

    if track == "program":
        deadline = program_deadline_score(item.get("date") or item.get("deadline"), now=None)
        topic = skill_fit(body, profile_skills, org)
        matched_skills, missing_skills = topic["matched"], topic["missing"]
        raw = combine([
            (deadline["score"], _PROGRAM_WEIGHTS["deadline"]),
            (None if topic["unknown"] else topic["score"], _PROGRAM_WEIGHTS["topic"]),
        ])
        reasons.append(deadline["note"])
        if not topic["unknown"]:
            reasons.append(f"matches {len(matched_skills)}/{len(matched_skills) + len(missing_skills)} stated eligibility keywords")
        confidence = 1.0
    else:
        skill = skill_fit(body, profile_skills, org)
        role = role_fit(title, target_roles)
        level = level_fit(title, description)
        matched_skills, missing_skills = skill["matched"], skill["missing"]

        # Skill never redistributes -- it's the dominant term, and handing its
        # weight to title/level lets a content-free "AI Engineer Intern"
        # posting (both maxed trivially) outrank a posting that actually
        # named and matched several real skills.
        skill_term = 0.5 if skill["unknown"] else skill["score"]
        raw = combine([
            (skill_term, _INDUSTRY_WEIGHTS["skill"]),
            (role["score"], _INDUSTRY_WEIGHTS["role"]),
            (level["score"], _INDUSTRY_WEIGHTS["level"]),
            (recency_score(item.get("date")), _INDUSTRY_WEIGHTS["recency"]),
        ])

        if skill["unknown"]:
            reasons.append("posting names no recognizable skills")
        else:
            reasons.append(
                f"matches {len(matched_skills)}/{len(matched_skills) + len(missing_skills)} "
                f"of the skills asked for" + (f" ({', '.join(matched_skills[:5])})" if matched_skills else "")
            )
            if missing_skills:
                reasons.append(f"missing {', '.join(missing_skills[:4])}")
        if role["best"] and role["score"] >= 0.35:
            reasons.append(f"title matches your \"{role['best']}\" target")
        elif role["score"] == 0:
            reasons.append("title is not one of your target roles")
        reasons.append(level["note"])

        confidence = description_confidence(description, track)
        if confidence < 1:
            raw = shrink_to_neutral(raw, confidence)
            reasons.append("short posting — limited evidence" if description else "no description available")

    if flags:
        raw *= 0.25
        reasons.insert(0, f"⚠ {'; '.join(flags)}")

    raw = max(0.0, min(1.0, raw))
    score_0_10 = round(raw * 10, 1)
    return {
        "score": score_0_10,
        "tier": _tier_of(score_0_10),
        "match_reasons": [r for r in reasons if not r.startswith("missing") and not r.startswith("⚠")],
        "concerns": [r for r in reasons if r.startswith("missing") or r.startswith("⚠")],
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "flags": flags,
    }
