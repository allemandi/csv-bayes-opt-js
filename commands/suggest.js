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

// This function formats prediction output based on user requirements.
function formatPrediction(model, target, pred, chosenOutcome) {
  if (pred.type === "number") {
    const precision = model.metadata[target].precision || 0;
    const mu = pred.mu.toFixed(precision);
    const low = (pred.mu - 1.96 * pred.sd).toFixed(precision);
    const high = (pred.mu + 1.96 * pred.sd).toFixed(precision);
    return `${target} prediction: ${mu} (95% CI: ${low} to ${high})`;
  }
  const outcome = chosenOutcome || (pred.ranked[0] ? pred.ranked[0].label : "n/a");
  const found = pred.ranked.find((r) => r.label === outcome);
  const prob = found ? (100 * found.probability).toFixed(1) : "0.0";
  return `${prob}% likelihood of ${outcome}.`;
}

// This function asks for known inputs, optimizes unknowns, and logs a final suggestion.
async function runInteractiveSuggest(model, maximizeCol, chosenOutcome, sessionLog, dataDir) {
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

  const bestObserved = model.metadata[maximizeCol].originalType === "number"
    ? Math.max(...model.cleanedData.map((r) => Number(r[maximizeCol])))
    : 0;
  const candidates = [];
  for (let i = 0; i < 2000; i += 1) {
    const encoded = randomEncodedCandidate(model);
    applyFixedValuesToEncoded(model, encoded, fixed, skippedColumns);
    candidates.push(encoded);
  }

  const best = rankCandidatesByEI(model, maximizeCol, candidates, bestObserved, chosenOutcome)[0];
  const decoded = decodeUserRow(best.encoded, model, [maximizeCol]);
  const predictionSummary = formatPrediction(model, maximizeCol, best.pred, chosenOutcome);

  const sessionResult = {
    timestamp: new Date().toISOString(),
    maximize: maximizeCol,
    targetOutcome: chosenOutcome,
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

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  let maximizeCol = maximizeArg;
  if (!maximizeCol) {
    console.log(`Available columns: ${model.userColumns.join(", ")}`);
    maximizeCol = (await askQuestion(rl, "Pick a target column: ")).trim();
  }
  if (!model.userColumns.includes(maximizeCol)) {
    rl.close();
    throw new Error(`Unknown column: ${maximizeCol}`);
  }

  let chosenOutcome = null;
  if (model.metadata[maximizeCol].originalType === "number") {
    const prompt = `You selected ${maximizeCol} — the optimizer will find the input combination most likely to produce the highest value. Press enter to confirm or type a different column name. `;
    const confirm = (await askQuestion(rl, prompt)).trim();
    if (confirm !== "") {
      maximizeCol = confirm;
      if (!model.userColumns.includes(maximizeCol)) {
        rl.close();
        throw new Error(`Unknown column: ${maximizeCol}`);
      }
      // Recursively handle if they changed it, but for simplicity let's just proceed with one change
      if (model.metadata[maximizeCol].originalType === "text") {
        const cats = model.encoding.text[maximizeCol].categories;
        console.log(`You selected ${maximizeCol} — which outcome do you want to make most likely?`);
        console.log(`Options: ${cats.join(", ")}`);
        chosenOutcome = (await askQuestion(rl, "Choice: ")).trim();
      }
    }
  } else {
    const cats = model.encoding.text[maximizeCol].categories;
    console.log(`You selected ${maximizeCol} — which outcome do you want to make most likely?`);
    console.log(`Options: ${cats.join(", ")}`);
    chosenOutcome = (await askQuestion(rl, "Choice: ")).trim();
  }
  rl.close();

  const bestObserved = model.metadata[maximizeCol].originalType === "number"
    ? Math.max(...model.cleanedData.map((r) => Number(r[maximizeCol])))
    : 0;
  const candidates = [];
  for (let i = 0; i < 2000; i += 1) candidates.push(randomEncodedCandidate(model));
  const top = rankCandidatesByEI(model, maximizeCol, candidates, bestObserved, chosenOutcome).slice(0, 3);

  console.log("\nTop 3 automatic suggestions:");
  top.forEach((item, idx) => {
    const decoded = decodeUserRow(item.encoded, model, [maximizeCol]);
    console.log(`\n#${idx + 1}`);
    console.log(decoded);
    console.log(formatPrediction(model, maximizeCol, item.pred, chosenOutcome));
  });

  const sessionLog = [];
  await runInteractiveSuggest(model, maximizeCol, chosenOutcome, sessionLog, dataDir);
}

module.exports = { runSuggest };
