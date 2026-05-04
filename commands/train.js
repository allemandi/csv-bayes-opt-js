const fs = require("fs");
const path = require("path");
const {
  ensureDirectory,
  loadCsvRecords,
  inferColumnTypes,
  printColumnSummary,
  cleanAndImpute,
  handleOutliersInteractive,
  fitAndTransform,
  trainInternalModel,
  validateModel,
  printValidationReport,
  permutationImportance,
} = require("../utils/model-utils");

// This function runs the full training pipeline and saves a model JSON file.
async function runTrain(csvPath, outputPath) {
  const records = loadCsvRecords(csvPath);
  let metadata = inferColumnTypes(records);
  const originalUserColumns = Object.keys(metadata);
  printColumnSummary(metadata, originalUserColumns);

  const cleaned = cleanAndImpute(records, metadata);
  metadata = cleaned.metadata;
  const postOutlierRows = await handleOutliersInteractive(cleaned.cleanedRows, metadata, cleaned.userColumns);
  if (!postOutlierRows.length) throw new Error("All rows were removed during outlier handling.");

  const transformed = fitAndTransform(postOutlierRows, metadata);
  console.log(`\nStarting model training on ${transformed.encodedRows.length} rows and ${Object.keys(metadata).length} columns...`);
  const model = trainInternalModel(transformed.encodedRows, postOutlierRows, transformed.encoding, metadata, 0.35);
  model.userColumns = cleaned.userColumns;
  model.internalColumns = cleaned.internalColumns;
  model.companionMap = cleaned.companionMap;
  model.fillRules = cleaned.fillRules;
  model.cleanedData = postOutlierRows.map((r) => {
    const out = {};
    for (const c of cleaned.userColumns) out[c] = r[c];
    return out;
  });
  model.encodingDetails = {
    oneHot: transformed.encoding.text,
    scaling: transformed.encoding.numeric,
    orderedFeatures: transformed.encoding.orderedFeatureKeys,
  };

  const validationMode = transformed.encodedRows.length >= 20 ? "split" : "loo";
  const validation = validateModel(model, transformed.encodedRows, postOutlierRows, validationMode);
  model.validation = { mode: validationMode, perColumn: validation };
  printValidationReport(validation);

  const lowR2 = Object.entries(validation).filter(([, m]) => m.r2 < 0.5).map(([c]) => c);
  if (lowR2.length) console.warn(`\nWARNING: Low R2 columns (<0.5): ${lowR2.join(", ")}`);

  const importance = permutationImportance(model, transformed.encodedRows, postOutlierRows);
  model.featureImportance = importance;
  console.log("\nColumn influence ranking:");
  for (const row of importance) console.log(`${row.rank}. ${row.column} (${row.importanceScore.toFixed(6)})`);

  ensureDirectory(path.dirname(outputPath));
  fs.writeFileSync(outputPath, JSON.stringify(model, null, 2), "utf8");
  console.log(`\nModel saved to ${outputPath}`);
}

module.exports = { runTrain };
