/*
  DecodeLabs | AI | Project 3 - Tech Stack Recommender (Web version)
  Author: Aakarsh Kumar
  Pipeline: Ingestion -> Scoring (TF-IDF + cosine) -> Sorting -> Filtering (Top 3)
*/
const MIN_SKILLS = 3;
const TOP_N = 3;

// Vocabulary alignment: different names for the same skill map to one key
const ALIASES = {
  "ml": "machine learning", "ai": "machine learning", "dl": "deep learning",
  "cloud": "cloud computing", "k8s": "kubernetes", "js": "javascript",
  "ts": "typescript", "node": "node.js", "nodejs": "node.js", "cicd": "ci/cd",
  "dsa": "data structures", "ds": "data structures", "powerbi": "power bi",
  "ui": "ui design", "ux": "user research", "llm": "llms", "api": "apis",
  "frontend": "html", "web design": "html", "cyber security": "security",
};
const normalize = (s) => {
  const t = s.toLowerCase().trim().replace(/\s+/g, " ");
  return ALIASES[t] || t;
};

// ---------- math ----------
const N = ROLES.length;
const df = {};
ROLES.forEach((r) => new Set(r.skills).forEach((s) => (df[s] = (df[s] || 0) + 1)));
const IDF = Object.fromEntries(Object.entries(df).map(([s, c]) => [s, Math.log(N / c)]));
const VOCAB = Object.keys(IDF).sort();

function tfidf(skills) {            // TF = count / total terms; weight = TF x IDF
  const counts = {};
  skills.forEach((s) => (counts[s] = (counts[s] || 0) + 1));
  const vec = {};
  for (const s in counts) vec[s] = (counts[s] / skills.length) * (IDF[s] || 0);
  return vec;
}

function cosine(a, b) {             // (A . B) / (||A|| ||B||), 0 for zero vectors
  let dot = 0, na = 0, nb = 0;
  for (const k in a) { dot += a[k] * (b[k] || 0); na += a[k] ** 2; }
  for (const k in b) nb += b[k] ** 2;
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

function recommend(userSkills) {
  const userVec = tfidf(userSkills);                                  // 1. ingestion
  return ROLES.map((r) => ({                                          // 2. scoring
    role: r.role,
    score: cosine(userVec, tfidf(r.skills)),
    matched: r.skills.filter((s) => userSkills.includes(s)),
    missing: r.skills.filter((s) => !userSkills.includes(s)),
  }))
    .sort((a, b) => b.score - a.score)                                // 3. sorting
    .slice(0, TOP_N);                                                 // 4. filtering
}

// ---------- UI ----------
const $ = (id) => document.getElementById(id);
const selected = new Set();

function msg(text) { $("msg").textContent = text; }

function renderChips() {
  $("chips").innerHTML = VOCAB.map(
    (s) => `<button type="button" class="chip ${selected.has(s) ? "on" : ""}" data-skill="${s}">${s}</button>`
  ).join("");
  $("count").textContent = selected.size;
}

function addSkill(raw) {
  const s = normalize(raw);
  if (!s) return;
  if (!(s in IDF)) { msg(`"${raw}" is not in the dataset. Pick from the chips below.`); return; }
  selected.add(s);
  msg("");
  renderChips();
}

function show(title, html) {
  $("results-title").textContent = title;
  $("list").innerHTML = html;
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth" });
}

$("add-form").addEventListener("submit", (e) => {
  e.preventDefault();
  $("skill-input").value.split(",").forEach(addSkill);
  $("skill-input").value = "";
});

$("chips").addEventListener("click", (e) => {
  const s = e.target.dataset.skill;
  if (!s) return;
  selected.has(s) ? selected.delete(s) : selected.add(s);
  renderChips();
});

$("go").addEventListener("click", () => {
  if (selected.size < MIN_SKILLS) { msg(`Please select at least ${MIN_SKILLS} skills (you have ${selected.size}).`); return; }
  msg("");
  show(`Top ${TOP_N} career paths for you`, recommend([...selected]).map((r) => `
    <div class="item">
      <h3>${r.role} <b>${(r.score * 100).toFixed(1)}% match</b></h3>
      <div class="bar"><i style="width:${(r.score * 100).toFixed(1)}%"></i></div>
      <p class="have">You already have: ${r.matched.join(", ") || "-"}</p>
      <p class="learn">Skills to learn: ${r.missing.join(", ") || "-"}</p>
    </div>`).join(""));
});

// Cold-start fallback: most popular roles when we know nothing about the user
$("trending").addEventListener("click", () => {
  const top = [...ROLES].sort((a, b) => b.popularity - a.popularity).slice(0, TOP_N);
  show("Trending roles (cold-start fallback)", top.map((r) => `
    <div class="item">
      <h3>${r.role} <b>demand ${r.popularity * 10}%</b></h3>
      <p class="learn">Core skills: ${r.skills.join(", ")}</p>
    </div>`).join(""));
});

$("clear").addEventListener("click", () => { selected.clear(); msg(""); $("results").hidden = true; renderChips(); });

renderChips();