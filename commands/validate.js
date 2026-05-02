const fs = require("fs");
const { printValidationReport, mean } = require("../utils/model-utils");

// This function loads a model file and prints trust metrics without retraining.
function runValidate(modelPath) {
  if (!fs.existsSync(modelPath)) throw new Error(`Model file not found: ${modelPath}`);
  const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));
  if (!model.validation || !model.featureImportance) throw new Error("Model file is missing validation or feature importance data.");

  printValidationReport(model.validation.perColumn);
  console.log("\nColumn influence ranking:");
  for (const row of model.featureImportance) console.log(`${row.rank}. ${row.column} (${Number(row.importanceScore).toFixed(6)})`);

  const avgR2 = mean(Object.values(model.validation.perColumn).map((m) => Number(m.r2) || 0));
  const trust = avgR2 >= 0.75 ? "Trustworthy" : avgR2 >= 0.5 ? "Use with caution" : "Not trustworthy";
  console.log(`\nOverall verdict: ${trust}`);
}

module.exports = { runValidate };
