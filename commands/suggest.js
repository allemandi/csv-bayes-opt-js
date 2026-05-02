const fs = require("fs");
const readline = require("readline");
const {
  askQuestion,
  decodeUserRow,
  randomEncodedCandidate,
  applyFixedValuesToEncoded,
  rankCandidatesByEI,
  persistSuggestions,
} = require("../utils/model-utils");

// This function formats categorical probabilities in ranked plain-English form.
function formatRankedCategories(ranked) {
  if (!ranked.length) return "No category probabilities available.";
  const [top, ...rest] = ranked;
  const topPct = `${(100 * top.probability).toFixed(1)}%`;
  if (!rest.length) return `Most likely outcome: ${top.label} (${topPct}).`;
  const alt = rest.map((r) => `${r.label} (${(100 * r.probability).toFixed(1)}%)`).join(", ");
  return `Most likely outcome: ${top.label} (${topPct}). Alternatives: ${alt}.`;
}

// This function detects if a target column is binary-like for percentage confidence messaging.
function isBinaryCategoricalTarget(model, maximizeCol) {
  if (!model.encoding.text[maximizeCol]) return false;
  return model.encoding.text[maximizeCol].categories.length === 2;
}

// This function asks for known inputs, optimizes unknowns, and logs a final suggestion.
async function runInteractiveSuggest(model, maximizeCol, sessionLog, dataDir) {
  const inputColumns = model.userColumns.filter((c) => c !== maximizeCol);
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const fixed = {};
  const skippedColumns = new Set();

  console.log("\nInteractive mode: press Enter for unknown values.");
  for (const c of inputColumns) {
    const prompt = model.metadata[c].originalType === "number"
      ? `${c} (number, blank=optimize): `
      : `${c} (text, options: ${model.encoding.text[c] ? model.encoding.text[c].categories.join(", ") : "n/a"}, blank=optimize): `;
    const ans = (await askQuestion(rl, prompt)).trim();
    if (ans === "") {
      skippedColumns.add(c);
      continue;
    }
    if (model.metadata[c].originalType === "number") {
      const n = Number(ans);
      if (!Number.isFinite(n)) {
        console.log(`Skipping invalid numeric value for ${c}.`);
        skippedColumns.add(c);
        continue;
      }
      fixed[c] = n;
      continue;
    }
    const allowed = model.encoding.text[c].categories;
    if (!allowed.includes(ans)) {
      console.log(`Skipping unknown category for ${c}.`);
      skippedColumns.add(c);
      continue;
    }
    fixed[c] = ans;
  }
  rl.close();

  const bestObserved = Math.max(...model.cleanedData.map((r) => Number(r[maximizeCol])));
  const candidates = [];
  for (let i = 0; i < 2000; i += 1) {
    const encoded = randomEncodedCandidate(model);
    applyFixedValuesToEncoded(model, encoded, fixed, skippedColumns);
    candidates.push(encoded);
  }

  const best = rankCandidatesByEI(model, maximizeCol, candidates, bestObserved)[0];
  const decoded = decodeUserRow(best.encoded, model, [maximizeCol]);
  let predictionSummary = "";
  if (best.pred.type === "number") {
    const low = best.pred.mu - 1.96 * best.pred.sd;
    const high = best.pred.mu + 1.96 * best.pred.sd;
    predictionSummary = `${maximizeCol} prediction: ${best.pred.mu.toFixed(4)} (95% CI: ${low.toFixed(4)} to ${high.toFixed(4)})`;
  } else if (isBinaryCategoricalTarget(model, maximizeCol)) {
    const top = best.pred.ranked[0];
    predictionSummary = `${(100 * top.probability).toFixed(1)}% likelihood of ${top.label}.`;
  } else {
    predictionSummary = formatRankedCategories(best.pred.ranked);
  }

  const sessionResult = {
    timestamp: new Date().toISOString(),
    maximize: maximizeCol,
    fixedInputs: fixed,
    suggestion: decoded,
  };
  if (best.pred.type === "number") {
    const low = best.pred.mu - 1.96 * best.pred.sd;
    const high = best.pred.mu + 1.96 * best.pred.sd;
    sessionResult.predictedTarget = best.pred.mu;
    sessionResult.confidence95 = [low, high];
  } else {
    sessionResult.mostLikely = best.pred.ranked[0];
    sessionResult.probabilities = best.pred.ranked;
  }
  sessionLog.push(sessionResult);

  console.log("\nFinal suggestion:");
  console.log(decoded);
  console.log(predictionSummary);

  const file = persistSuggestions(sessionLog, dataDir);
  if (file) console.log(`Saved session suggestions to ${file}`);
}

// This function runs automatic top suggestions and then interactive optimization.
async function runSuggest(modelPath, maximizeArg, dataDir) {
  if (!fs.existsSync(modelPath)) throw new Error(`Model file not found: ${modelPath}`);
  const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));
  if (!Array.isArray(model.userColumns) || !model.userColumns.length) {
    throw new Error("Model file is missing user column metadata. Retrain with the latest version.");
  }
  let maximizeCol = maximizeArg;
  if (!maximizeCol) {
    console.log(`Available columns to maximize: ${model.userColumns.join(", ")}`);
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    maximizeCol = (await askQuestion(rl, "Pick one column to maximize: ")).trim();
    rl.close();
  }
  if (!model.userColumns.includes(maximizeCol)) throw new Error(`Unknown column for --maximize: ${maximizeCol}`);
  if (model.metadata[maximizeCol].originalType === "number") {
    console.log(`Target '${maximizeCol}' is numeric: optimizer uses Expected Improvement.`);
  } else {
    const classCount = model.encoding.text[maximizeCol].categories.length;
    console.log(`Target '${maximizeCol}' is categorical (${classCount} classes): optimizer maximizes top-class probability.`);
  }

  const bestObserved = model.metadata[maximizeCol].originalType === "number"
    ? Math.max(...model.cleanedData.map((r) => Number(r[maximizeCol])))
    : 0;
  const candidates = [];
  for (let i = 0; i < 2000; i += 1) candidates.push(randomEncodedCandidate(model));
  const top = rankCandidatesByEI(model, maximizeCol, candidates, bestObserved).slice(0, 3);

  console.log("\nTop 3 automatic suggestions:");
  top.forEach((item, idx) => {
    const decoded = decodeUserRow(item.encoded, model, [maximizeCol]);
    console.log(`\n#${idx + 1}`);
    console.log(decoded);
    if (item.pred.type === "number") {
      const low = item.pred.mu - 1.96 * item.pred.sd;
      const high = item.pred.mu + 1.96 * item.pred.sd;
      console.log(`${maximizeCol}: ${item.pred.mu.toFixed(4)} (95% CI: ${low.toFixed(4)} to ${high.toFixed(4)})`);
    } else if (isBinaryCategoricalTarget(model, maximizeCol)) {
      const topOutcome = item.pred.ranked[0];
      console.log(`${(100 * topOutcome.probability).toFixed(1)}% likelihood of ${topOutcome.label}.`);
    } else {
      console.log(formatRankedCategories(item.pred.ranked));
    }
  });

  const sessionLog = [];
  await runInteractiveSuggest(model, maximizeCol, sessionLog, dataDir);
}

module.exports = { runSuggest };
