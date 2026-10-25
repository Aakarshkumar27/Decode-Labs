/*
  DecodeLabs | AI | Project 2 - Data Classification Using AI (Web version)
  Author: Aakarsh Kumar

  Pipeline (IPO):
    INPUT   -> Iris dataset (data.js) + StandardScaler
    PROCESS -> shuffled, stratified train/test split + KNN (from scratch)
    OUTPUT  -> accuracy, confusion matrix, precision / recall / F1, K-tuning chart
*/

// ---------- Helpers ----------
const $ = (id) => document.getElementById(id);

// Seeded random generator so results are reproducible (like random_state=42)
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------- Train / test split (shuffled + stratified) ----------
function trainTestSplit(data, testFraction, seed = 42) {
  const rand = mulberry32(seed);
  const train = [];
  const test = [];
  for (let c = 0; c < CLASS_NAMES.length; c++) {
    const group = shuffle(data.filter((r) => r[4] === c), rand);
    const nTest = Math.round(group.length * testFraction);
    test.push(...group.slice(0, nTest));
    train.push(...group.slice(nTest));
  }
  return { train: shuffle(train, rand), test: shuffle(test, rand) };
}

// ---------- StandardScaler: mean = 0, variance = 1 (fit on TRAIN only) ----------
function fitScaler(rows) {
  const n = rows.length;
  const mean = [0, 1, 2, 3].map((j) => rows.reduce((s, r) => s + r[j], 0) / n);
  const std = [0, 1, 2, 3].map((j) =>
    Math.sqrt(rows.reduce((s, r) => s + (r[j] - mean[j]) ** 2, 0) / n) || 1
  );
  return { mean, std };
}

function transform(features, scaler) {
  return features.map((v, j) => (v - scaler.mean[j]) / scaler.std[j]);
}

// ---------- K-Nearest Neighbors ----------
class KNN {
  constructor(k) { this.k = k; }

  fit(X, y) { this.X = X; this.y = y; return this; }

  predictOne(x) {
    const dists = this.X.map((p, i) => ({
      d: Math.sqrt(p.reduce((s, v, j) => s + (v - x[j]) ** 2, 0)),
      label: this.y[i],
    }));
    dists.sort((a, b) => a.d - b.d);
    const neighbors = dists.slice(0, this.k);

    // Majority vote; ties go to the class with the closest neighbor
    const votes = new Array(CLASS_NAMES.length).fill(0);
    neighbors.forEach((n) => votes[n.label]++);
    const max = Math.max(...votes);
    return neighbors.find((n) => votes[n.label] === max).label;
  }

  predict(X) { return X.map((x) => this.predictOne(x)); }
}

// ---------- Metrics ----------
function confusionMatrix(yTrue, yPred) {
  const n = CLASS_NAMES.length;
  const m = Array.from({ length: n }, () => new Array(n).fill(0));
  yTrue.forEach((t, i) => m[t][yPred[i]]++);
  return m;
}

function classMetrics(cm) {
  return CLASS_NAMES.map((_, c) => {
    const tp = cm[c][c];
    const fp = cm.reduce((s, row, r) => s + (r === c ? 0 : row[c]), 0);
    const fn = cm[c].reduce((s, v, p) => s + (p === c ? 0 : v), 0);
    const support = tp + fn;
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1, support };
  });
}

// ---------- Pipeline ----------
let state = null; // holds the latest trained model so the predictor can use it

function runPipeline() {
  const testFraction = Number($("split").value) / 100;
  const k = Number($("k").value);
  const useScaling = $("scale").checked;

  const { train, test } = trainTestSplit(IRIS, testFraction);
  const scaler = useScaling ? fitScaler(train) : { mean: [0, 0, 0, 0], std: [1, 1, 1, 1] };

  const Xtrain = train.map((r) => transform(r.slice(0, 4), scaler));
  const ytrain = train.map((r) => r[4]);
  const Xtest = test.map((r) => transform(r.slice(0, 4), scaler));
  const ytest = test.map((r) => r[4]);

  const model = new KNN(k).fit(Xtrain, ytrain);
  const preds = model.predict(Xtest);

  const cm = confusionMatrix(ytest, preds);
  const metrics = classMetrics(cm);
  const total = ytest.length;
  const correct = cm.reduce((s, row, i) => s + row[i], 0);
  const accuracy = correct / total;
  const weightedF1 = metrics.reduce((s, m) => s + m.f1 * m.support, 0) / total;

  // Error rate for every K (tuning the engine)
  const errors = [];
  for (let kk = 1; kk <= 25; kk++) {
    const p = new KNN(kk).fit(Xtrain, ytrain).predict(Xtest);
    errors.push(1 - p.filter((v, i) => v === ytest[i]).length / total);
  }

  state = { model, scaler };
  renderResults({ train, test, accuracy, weightedF1, cm, metrics, errors, k });
}

// ---------- Rendering ----------
function renderPreview() {
  const head = `<tr><th>#</th>${FEATURE_NAMES.map((n) => `<th>${n}</th>`).join("")}<th>Species</th></tr>`;
  const rows = IRIS.filter((_, i) => i % 15 === 0 || i === 149)
    .map((r) => {
      const idx = IRIS.indexOf(r) + 1;
      return `<tr><td>${idx}</td>${r.slice(0, 4).map((v) => `<td>${v}</td>`).join("")}<td>${CLASS_NAMES[r[4]]}</td></tr>`;
    })
    .join("");
  $("preview").innerHTML = head + rows;
}

function renderResults({ train, test, accuracy, weightedF1, cm, metrics, errors, k }) {
  $("split-info").textContent =
    `Training samples: ${train.length} | Testing samples: ${test.length} (shuffled, class balance kept)`;

  $("results").hidden = false;
  $("acc").textContent = (accuracy * 100).toFixed(2) + "%";
  $("f1").textContent = weightedF1.toFixed(4);

  // Confusion matrix
  let html = `<tr><th>Actual \\ Predicted</th>${CLASS_NAMES.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  cm.forEach((row, i) => {
    html += `<tr><th>${CLASS_NAMES[i]}</th>${row
      .map((v, j) => `<td class="${i === j ? "diag" : v > 0 ? "off" : ""}">${v}</td>`)
      .join("")}</tr>`;
  });
  $("cm").innerHTML = html;

  // Per-class report
  let rep = `<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>`;
  metrics.forEach((m, i) => {
    rep += `<tr><td>${CLASS_NAMES[i]}</td><td>${m.precision.toFixed(3)}</td><td>${m.recall.toFixed(3)}</td><td>${m.f1.toFixed(3)}</td><td>${m.support}</td></tr>`;
  });
  $("report").innerHTML = rep;

  drawKChart(errors, k);
}

// Simple canvas line chart: error rate vs K
function drawKChart(errors, selectedK) {
  const canvas = $("k-chart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 50, r: 20, t: 20, b: 40 };
  ctx.clearRect(0, 0, W, H);

  const maxErr = Math.max(0.1, ...errors) * 1.15;
  const x = (kk) => pad.l + ((kk - 1) / (errors.length - 1)) * (W - pad.l - pad.r);
  const y = (e) => H - pad.b - (e / maxErr) * (H - pad.t - pad.b);

  // Axes and grid
  ctx.strokeStyle = "#d5dee8";
  ctx.fillStyle = "#5b6b7c";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const e = (maxErr / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y(e)); ctx.lineTo(W - pad.r, y(e)); ctx.stroke();
    ctx.fillText((e * 100).toFixed(0) + "%", pad.l - 6, y(e) + 4);
  }
  ctx.textAlign = "center";
  [1, 5, 10, 15, 20, 25].forEach((kk) => ctx.fillText(kk, x(kk), H - pad.b + 18));
  ctx.fillText("K value", W / 2, H - 6);

  // Error line
  ctx.strokeStyle = "#0b3c6f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  errors.forEach((e, i) => (i ? ctx.lineTo(x(i + 1), y(e)) : ctx.moveTo(x(1), y(e))));
  ctx.stroke();

  // Points
  errors.forEach((e, i) => {
    ctx.beginPath();
    ctx.arc(x(i + 1), y(e), 3, 0, Math.PI * 2);
    ctx.fillStyle = "#0b3c6f";
    ctx.fill();
  });

  // Highlight the selected K and the best K
  const best = errors.indexOf(Math.min(...errors)) + 1;
  ctx.strokeStyle = "#e8590c";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x(selectedK), y(errors[selectedK - 1]), 8, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "#e8590c";
  ctx.textAlign = "left";
  ctx.fillText(`Selected K = ${selectedK}  |  lowest error at K = ${best}`, pad.l + 8, pad.t + 6);
}

// ---------- Predict a new flower ----------
function predictFlower() {
  const values = ["f0", "f1in", "f2", "f3"].map((id) => parseFloat($(id).value));
  if (values.some(Number.isNaN)) {
    $("prediction").textContent = "Please fill in all four measurements.";
    return;
  }
  const scaled = transform(values, state.scaler);
  const label = state.model.predictOne(scaled);
  $("prediction").innerHTML = `Predicted species: <em>Iris ${CLASS_NAMES[label]}</em>`;
}

// ---------- Events ----------
$("split").addEventListener("input", () => ($("split-label").textContent = $("split").value + "%"));
$("k").addEventListener("input", () => ($("k-label").textContent = $("k").value));
$("train-btn").addEventListener("click", runPipeline);
$("predict-btn").addEventListener("click", predictFlower);

renderPreview();
runPipeline(); // train once on load so the page is never empty