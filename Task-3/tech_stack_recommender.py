"""
DecodeLabs | Artificial Intelligence | Project 3
AI Recommendation Logic: Tech Stack Recommender (Content-Based Filtering)

Author : Aakarsh Kumar
Batch  : 2026

Pipeline:
    1. INGESTION : take at least 3 skills from the user
    2. SCORING   : TF-IDF vectors + cosine similarity against every job role
    3. SORTING   : rank roles by score (descending)
    4. FILTERING : keep only the Top-N (default 3)

Run:  python tech_stack_recommender.py      (needs raw_skills.csv in the same folder)
No external libraries required.
"""

import csv
import difflib
import math
import os
from collections import Counter

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_skills.csv")
MIN_SKILLS = 3
TOP_N = 3

# Vocabulary alignment: different names for the same skill map to one key
ALIASES = {
    "ml": "machine learning", "ai": "machine learning", "dl": "deep learning",
    "cloud": "cloud computing", "k8s": "kubernetes", "js": "javascript",
    "ts": "typescript", "node": "node.js", "nodejs": "node.js", "cicd": "ci/cd",
    "dsa": "data structures", "ds": "data structures", "powerbi": "power bi",
    "ui": "ui design", "ux": "user research", "llm": "llms", "api": "apis",
    "frontend": "html", "web design": "html", "cyber security": "security",
}


def normalize(skill: str) -> str:
    s = " ".join(skill.lower().strip().split())
    return ALIASES.get(s, s)


# ---------------------------------------------------------------- data
def load_roles(path: str = CSV_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [
            {
                "role": row["role"],
                "skills": [normalize(s) for s in row["skills"].split(";")],
                "popularity": int(row["popularity"]),
            }
            for row in csv.DictReader(f)
        ]


# ---------------------------------------------------------------- math
def build_idf(roles: list[dict]) -> dict[str, float]:
    """IDF = log(total documents / documents containing the term)."""
    n = len(roles)
    df = Counter(skill for r in roles for skill in set(r["skills"]))
    return {skill: math.log(n / count) for skill, count in df.items()}


def tfidf_vector(skills: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF = count of term / total terms. Weight = TF x IDF (sparse vector)."""
    counts = Counter(skills)
    total = len(skills)
    return {s: (c / total) * idf.get(s, 0.0) for s, c in counts.items()}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """cos(theta) = (A . B) / (||A|| ||B||); returns 0 for zero vectors."""
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ---------------------------------------------------------------- pipeline
def recommend(user_skills: list[str], roles: list[dict], idf: dict, top_n: int = TOP_N):
    user_vec = tfidf_vector(user_skills, idf)                        # step 1 (ingestion)
    scored = []
    for r in roles:                                                  # step 2 (scoring)
        score = cosine_similarity(user_vec, tfidf_vector(r["skills"], idf))
        scored.append({
            "role": r["role"],
            "score": score,
            "matched": [s for s in r["skills"] if s in user_skills],
            "missing": [s for s in r["skills"] if s not in user_skills],
        })
    scored.sort(key=lambda x: x["score"], reverse=True)              # step 3 (sorting)
    return scored[:top_n]                                            # step 4 (filtering)


def trending(roles: list[dict], top_n: int = TOP_N) -> list[dict]:
    """Cold-start fallback: most popular roles when we know nothing about the user."""
    return sorted(roles, key=lambda r: r["popularity"], reverse=True)[:top_n]


# ---------------------------------------------------------------- interface
def show_results(results: list[dict], is_trending: bool = False) -> None:
    print("\n" + "=" * 60)
    print(f"TOP {len(results)} " + ("TRENDING ROLES" if is_trending else "CAREER PATHS FOR YOU"))
    print("=" * 60)
    for i, r in enumerate(results, 1):
        if is_trending:
            print(f"\n{i}. {r['role']}  -  demand {r['score']:.0%}")
            print(f"   Core skills: {', '.join(r['missing'])}")
            continue
        bar = "#" * round(r["score"] * 20)
        print(f"\n{i}. {r['role']}  -  {r['score']:.1%} match  [{bar:<20}]")
        print(f"   You already have : {', '.join(r['matched']) or '-'}")
        print(f"   Skills to learn  : {', '.join(r['missing']) or '-'}")


def main() -> None:
    roles = load_roles()
    idf = build_idf(roles)
    vocabulary = sorted(idf)

    print("=" * 60)
    print("  TECH STACK RECOMMENDER  |  DecodeLabs Project 3")
    print("=" * 60)
    print("Known skills:\n  " + ", ".join(vocabulary))
    print(f"\nEnter at least {MIN_SKILLS} skills separated by commas.")
    print("Type 'trending' for popular roles, or 'exit' to quit.")

    while True:
        raw = input("\nYour skills: ").strip()
        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "bye"}:
            print("Goodbye! Keep building.")
            break
        if raw.lower() == "trending":
            show_results([{"role": r["role"], "score": r["popularity"] / 10,
                           "matched": [], "missing": r["skills"]} for r in trending(roles)],
                         is_trending=True)
            continue

        entered = [normalize(s) for s in raw.split(",") if s.strip()]
        known = list(dict.fromkeys(s for s in entered if s in idf))   # dedupe, keep order
        for s in (s for s in entered if s not in idf):
            hint = difflib.get_close_matches(s, vocabulary, n=1)
            print(f"  '{s}' is not in the dataset" + (f" - did you mean '{hint[0]}'?" if hint else "."))

        if len(known) < MIN_SKILLS:
            print(f"  Need at least {MIN_SKILLS} valid skills (got {len(known)}). Try again.")
            continue

        show_results(recommend(known, roles, idf))


if __name__ == "__main__":
    main()