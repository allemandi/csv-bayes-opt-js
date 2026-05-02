const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { parse } = require("csv-parse/sync");
const dfd = require("danfojs-node");
const { Matrix, CholeskyDecomposition } = require("ml-matrix");
const { RandomForestClassifier, RandomForestRegression } = require("ml-random-forest/random-forest.js");

// This function detects whether a value should be treated as missing.
function isBlank(value) {
  return value === null || value === undefined || String(value).trim() === "";
}

// This function computes the average value for a list of numbers.
function mean(values) {
  if (!values.length) return 0;
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

// This function computes mean absolute error between true and predicted values.
function mae(yTrue, yPred) {
  if (!yTrue.length) return 0;
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) total += Math.abs(yTrue[i] - yPred[i]);
  return total / yTrue.length;
}

// This function computes R-squared to estimate prediction quality.
function r2(yTrue, yPred) {
  if (!yTrue.length) return 0;
  const yMean = mean(yTrue);
  const ssRes = yTrue.reduce((acc, y, i) => acc + (y - yPred[i]) ** 2, 0);
  const ssTot = yTrue.reduce((acc, y) => acc + (y - yMean) ** 2, 0);
  if (ssTot === 0) return 1;
  return 1 - ssRes / ssTot;
}

// This function returns a shuffled copy of an input array.
function shuffled(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

// This function asks one interactive question and resolves with the answer.
function askQuestion(rl, question) {
  return new Promise((resolve) => rl.question(question, (answer) => resolve(answer)));
}

// This function ensures a directory exists before writing files into it.
function ensureDirectory(targetDir) {
  if (!fs.existsSync(targetDir)) fs.mkdirSync(targetDir, { recursive: true });
}

// This function loads CSV data rows from disk.
function loadCsvRecords(csvPath) {
  if (!fs.existsSync(csvPath)) throw new Error(`CSV file not found: ${csvPath}`);
  const raw = fs.readFileSync(csvPath, "utf8");
  const records = parse(raw, { columns: true, skip_empty_lines: true, relax_column_count: true, trim: true });
  if (!records.length) throw new Error("CSV has no data rows.");
  return records;
}

// This function infers numeric/text type for each original CSV column.
function inferColumnTypes(records) {
  const columns = Object.keys(records[0] || {});
  const metadata = {};
  for (const col of columns) {
    const values = records.map((r) => r[col]).filter((v) => !isBlank(v));
    let numericCount = 0;
    for (const v of values) {
      const n = Number(v);
      if (!Number.isNaN(n) && Number.isFinite(n)) numericCount += 1;
    }
    const numeric = values.length > 0 && numericCount / values.length >= 0.9;
    metadata[col] = { originalType: numeric ? "number" : "text", internal: false };
  }
  return metadata;
}

// This function prints an easy table of detected column types.
function printColumnSummary(metadata, userColumns) {
  const rows = userColumns.map((column) => ({ column, inferredType: metadata[column].originalType }));
  const frame = new dfd.DataFrame(rows);
  console.log("\nDetected columns:");
  frame.print();
}

// This function drops sparse rows and imputes missing values while creating internal missing flags.
function cleanAndImpute(records, metadata) {
  const userColumns = Object.keys(metadata).filter((c) => !metadata[c].internal);
  const normalized = records.map((row) => {
    const copy = {};
    for (const c of userColumns) copy[c] = isBlank(row[c]) ? null : row[c];
    return copy;
  });
  const baseDf = new dfd.DataFrame(normalized);
  const keptRows = [];
  for (let i = 0; i < baseDf.shape[0]; i += 1) {
    const rowVals = baseDf.iloc({ rows: [i] }).values[0];
    const blanks = rowVals.filter((v) => v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))).length;
    if (blanks <= userColumns.length / 2) {
      const rowObj = {};
      for (let j = 0; j < userColumns.length; j += 1) rowObj[userColumns[j]] = rowVals[j];
      keptRows.push(rowObj);
    }
  }
  if (!keptRows.length) throw new Error("No rows remain after dropping rows with >50% blank values.");
  const keptDf = new dfd.DataFrame(keptRows);
  const fillRules = {};
  for (const c of userColumns) {
    if (metadata[c].originalType === "number") {
      const numericSeries = keptDf[c].asType("float32");
      fillRules[c] = { fillValue: numericSeries.median(), strategy: "median" };
    } else {
      const stringSeries = keptDf[c].fillNa("__missing__");
      const counts = stringSeries.valueCounts();
      const mode = counts.index && counts.index.length ? String(counts.index[0]) : "__missing__";
      fillRules[c] = { fillValue: mode, strategy: "mode" };
    }
  }

  const filledColumns = [];
  const companionMap = {};
  for (const row of keptRows) {
    for (const c of userColumns) {
      const wasMissing = isBlank(row[c]);
      const missingFlagCol = `${c}_was_missing`;
      companionMap[c] = missingFlagCol;
      row[missingFlagCol] = wasMissing;
      if (wasMissing) {
        row[c] = fillRules[c].fillValue;
        if (!filledColumns.includes(c)) filledColumns.push(c);
      }
      row[c] = metadata[c].originalType === "number" ? Number(row[c]) : String(row[c]);
    }
  }

  const missingFlagMetadata = {};
  for (const c of userColumns) missingFlagMetadata[`${c}_was_missing`] = { originalType: "number", generated: true, internal: true };

  console.log(`\nDropped rows with >50% missing: ${records.length - keptRows.length}`);
  if (filledColumns.length) console.log(`Filled missing cells in columns: ${filledColumns.join(", ")}`);
  else console.log("No missing cells required imputation.");

  return {
    cleanedRows: keptRows,
    metadata: { ...metadata, ...missingFlagMetadata },
    fillRules,
    userColumns,
    internalColumns: Object.keys(missingFlagMetadata),
    companionMap,
  };
}

// This function reviews numeric outliers with the user and keeps or drops rows.
async function handleOutliersInteractive(rows, metadata, userColumns) {
  const numericCols = userColumns.filter((c) => metadata[c].originalType === "number");
  const df = new dfd.DataFrame(rows);
  const stats = numericCols.reduce((acc, c) => {
    const series = df[c].asType("float32");
    const stdValue = Number(series.std());
    acc[c] = { mean: Number(series.mean()), std: Number.isFinite(stdValue) ? stdValue : 0 };
    return acc;
  }, {});

  const outlierIndices = [];
  rows.forEach((row, idx) => {
    const hits = [];
    for (const c of numericCols) {
      const s = stats[c];
      if (!s || s.std === 0) continue;
      const v = Number(row[c]);
      if (Math.abs(v - s.mean) > 3 * s.std) hits.push(c);
    }
    if (hits.length) outlierIndices.push({ idx, hits });
  });

  if (!outlierIndices.length) {
    console.log("\nNo 3-sigma numeric outliers found.");
    return rows;
  }

  console.log(`\nFound ${outlierIndices.length} outlier rows.`);
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const keep = new Set(rows.map((_, i) => i));
  for (const out of outlierIndices) {
    const preview = {};
    for (const c of userColumns) preview[c] = rows[out.idx][c];
    console.log(`\nOutlier row ${out.idx}:`, preview);
    console.log(`Trigger columns: ${out.hits.join(", ")}`);
    const answer = (await askQuestion(rl, "Drop this row? [y/N]: ")).trim().toLowerCase();
    if (answer === "y" || answer === "yes") keep.delete(out.idx);
  }
  rl.close();
  return rows.filter((_, i) => keep.has(i));
}

// This function fits one-hot and min-max transforms and encodes all rows.
function fitAndTransform(rows, metadata) {
  const columns = Object.keys(metadata);
  const numericCols = columns.filter((c) => metadata[c].originalType === "number");
  const textCols = columns.filter((c) => metadata[c].originalType === "text");
  const df = new dfd.DataFrame(rows);
  const encoding = { numeric: {}, text: {}, orderedFeatureKeys: [], originalColumns: columns };

  let scaledDf = new dfd.DataFrame([]);
  if (numericCols.length) {
    const numericDf = new dfd.DataFrame(
      rows.map((row) => {
        const out = {};
        for (const c of numericCols) out[c] = Number(row[c]);
        return out;
      })
    );
    const scaler = new dfd.MinMaxScaler();
    scaler.fit(numericDf);
    const transformed = scaler.transform(numericDf);
    scaledDf = new dfd.DataFrame(transformed.values, { columns: numericCols });
    for (const c of numericCols) {
      const originalSeries = numericDf[c];
      encoding.numeric[c] = { min: Number(originalSeries.min()), max: Number(originalSeries.max()) };
      encoding.orderedFeatureKeys.push(c);
    }
  }

  let dummiesDf = new dfd.DataFrame([]);
  if (textCols.length) {
    const textDf = df.loc({ columns: textCols });
    dummiesDf = dfd.getDummies(textDf, { columns: textCols, prefixSeparator: "__" });
    for (const c of textCols) {
      const categories = dummiesDf.columns
        .filter((col) => col.startsWith(`${c}__`))
        .map((col) => col.slice(`${c}__`.length))
        .sort();
      encoding.text[c] = { categories };
      for (const cat of categories) encoding.orderedFeatureKeys.push(`${c}__${cat}`);
    }
  }

  const encodedDf = dummiesDf.shape[0] && scaledDf.shape[0]
    ? dfd.concat({ dfList: [scaledDf, dummiesDf], axis: 1 })
    : (scaledDf.shape[0] ? scaledDf : dummiesDf);
  const encodedRows = encodedDf.values.map((vals) => {
    const out = {};
    for (let i = 0; i < encodedDf.columns.length; i += 1) out[encodedDf.columns[i]] = Number(vals[i] || 0);
    for (const key of encoding.orderedFeatureKeys) if (!(key in out)) out[key] = 0;
    return out;
  });
  return { encodedRows, encoding };
}

// This function converts one raw row into encoded numeric feature space.
function encodeRow(row, encoding) {
  const out = {};
  for (const [c, cfg] of Object.entries(encoding.numeric)) {
    const v = Number(row[c]);
    const denom = cfg.max - cfg.min;
    out[c] = denom === 0 ? 0 : (v - cfg.min) / denom;
  }
  for (const [c, cfg] of Object.entries(encoding.text)) {
    const raw = String(row[c]);
    for (const cat of cfg.categories) out[`${c}__${cat}`] = raw === cat ? 1 : 0;
  }
  return out;
}

// This function decodes an encoded row into user-visible columns only.
function decodeUserRow(encoded, model, excludeColumns = []) {
  const excluded = new Set(excludeColumns);
  const out = {};
  for (const c of model.userColumns) {
    if (excluded.has(c)) continue;
    if (model.encoding.numeric[c]) {
      const { min, max } = model.encoding.numeric[c];
      out[c] = min + (max - min) * (Number(encoded[c]) || 0);
      continue;
    }
    if (model.encoding.text[c]) {
      const cats = model.encoding.text[c].categories;
      let bestCat = cats[0];
      let bestVal = -Infinity;
      for (const cat of cats) {
        const score = Number(encoded[`${c}__${cat}`]) || 0;
        if (score > bestVal) {
          bestVal = score;
          bestCat = cat;
        }
      }
      out[c] = bestCat;
    }
  }
  return out;
}

// This function creates per-target feature layouts for model training.
function buildTargetConfigs(encoding, metadata) {
  const configs = {};
  const allFeatures = encoding.orderedFeatureKeys;
  for (const c of Object.keys(metadata)) {
    if (encoding.numeric[c]) configs[c] = { type: "number", yKeys: [c], xKeys: allFeatures.filter((f) => f !== c) };
    else if (encoding.text[c]) {
      const yKeys = encoding.text[c].categories.map((cat) => `${c}__${cat}`);
      configs[c] = { type: "text", yKeys, xKeys: allFeatures.filter((f) => !f.startsWith(`${c}__`)) };
    }
  }
  return configs;
}

// This function builds X and Y matrices for one prediction target.
function makeDatasetForTarget(encodedRows, cfg) {
  const X = encodedRows.map((r) => cfg.xKeys.map((k) => Number(r[k] || 0)));
  const Y = encodedRows.map((r) => cfg.yKeys.map((k) => Number(r[k] || 0)));
  return { X, Y };
}

// This function computes the RBF kernel value between two vectors.
function rbfKernel(a, b, lengthScale) {
  const diff = Matrix.rowVector(a).sub(Matrix.rowVector(b));
  const sqNorm = diff.mmul(diff.transpose()).get(0, 0);
  return Math.exp(-sqNorm / (2 * lengthScale * lengthScale));
}

// This function constructs the GP training kernel matrix with noise on the diagonal.
function buildKernelMatrix(trainX, lengthScale, noiseVariance) {
  const n = trainX.length;
  const K = Matrix.zeros(n, n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      K.set(i, j, rbfKernel(trainX[i], trainX[j], lengthScale));
    }
  }
  const noiseI = Matrix.eye(n, n).mul(noiseVariance);
  return K.add(noiseI);
}

// This function fits one Gaussian Process model and returns serializable state.
function fitGaussianProcess(trainX, trainY, lengthScale, noiseVariance) {
  const K = buildKernelMatrix(trainX, lengthScale, noiseVariance);
  const chol = new CholeskyDecomposition(K);
  const yCol = Matrix.columnVector(trainY);
  const alpha = chol.solve(yCol);
  return {
    trainX,
    trainY,
    alpha: alpha.to1DArray(),
    kernelMatrix: K.to2DArray(),
    lengthScale,
    noiseVariance,
  };
}

// This function predicts GP mean and variance for one candidate input.
function predictGaussianProcess(gpState, xStar) {
  const kStar = Matrix.columnVector(gpState.trainX.map((row) => rbfKernel(row, xStar, gpState.lengthScale)));
  const alphaCol = Matrix.columnVector(gpState.alpha);
  const mean = kStar.transpose().mmul(alphaCol).get(0, 0);
  const chol = new CholeskyDecomposition(new Matrix(gpState.kernelMatrix));
  const solveKStar = chol.solve(kStar);
  const kxx = rbfKernel(xStar, xStar, gpState.lengthScale) + gpState.noiseVariance;
  const variance = Math.max(kxx - kStar.transpose().mmul(solveKStar).get(0, 0), 1e-12);
  return { mean, variance };
}

// This function predicts one target column value and confidence from a candidate row.
function predictTarget(model, target, encodedRow) {
  const cfg = model.targetConfigs[target];
  const train = model.trainedTargets[target];
  const x = cfg.xKeys.map((k) => Number(encodedRow[k] || 0));
  let meanVec = [];
  let stdVec = [];

  if (train.algorithm === "gp") {
    const predByDim = train.gpByDim.map((gpState) => predictGaussianProcess(gpState, x));
    meanVec = predByDim.map((p) => p.mean);
    stdVec = predByDim.map((p) => Math.sqrt(p.variance));
  } else if (train.algorithm === "rf") {
    if (cfg.type === "number") {
      const rf = RandomForestRegression.load(train.rfModel);
      meanVec = [Number(rf.predict([x])[0])];
      const residualVar = Number(train.residualVariance || 1e-6);
      stdVec = [Math.sqrt(Math.max(residualVar, 1e-12))];
    } else {
      const rf = RandomForestClassifier.load(train.rfModel);
      const classCount = train.classLabels.length;
      meanVec = train.classLabels.map((label) => Number(rf.predictProbability([x], label)[0] || 0));
      stdVec = new Array(classCount).fill(0.1);
    }
  }

  if (cfg.type === "number") {
    const scaledMu = meanVec[0];
    const scaledSd = stdVec[0];
    const scale = model.encoding.numeric[target].max - model.encoding.numeric[target].min;
    return {
      type: "number",
      mu: model.encoding.numeric[target].min + scaledMu * scale,
      sd: scaledSd * Math.abs(scale || 1),
    };
  }

  const scores = meanVec.map((s) => (Number.isFinite(s) ? s : 0));
  const maxScore = Math.max(...scores);
  const stable = scores.map((s) => s - maxScore);
  const expScores = stable.map((s) => Math.exp(s));
  const sumExp = expScores.reduce((acc, v) => acc + v, 0) || 1;
  const probs = expScores.map((v) => v / sumExp);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i += 1) if (probs[i] > probs[bestIdx]) bestIdx = i;
  const cats = model.encoding.text[target].categories;
  const ranked = cats.map((label, idx) => ({ label, probability: probs[idx] })).sort((a, b) => b.probability - a.probability);
  return { type: "text", label: cats[bestIdx], probs, ranked };
}

// This function trains target-wise Gaussian Process models for all columns.
function trainInternalModel(encodedRows, encoding, metadata, lengthScale) {
  const noiseVariance = 1e-6;
  const LARGE_DATASET_THRESHOLD = 400;
  const useRandomForestFallback = encodedRows.length >= LARGE_DATASET_THRESHOLD;
  const targetConfigs = buildTargetConfigs(encoding, metadata);
  const trainedTargets = {};
  for (const [target, cfg] of Object.entries(targetConfigs)) {
    const dataset = makeDatasetForTarget(encodedRows, cfg);
    if (!useRandomForestFallback) {
      const gpByDim = [];
      for (let d = 0; d < dataset.Y[0].length; d += 1) {
        gpByDim.push(
          fitGaussianProcess(
            dataset.X,
            dataset.Y.map((row) => row[d]),
            lengthScale,
            noiseVariance
          )
        );
      }
      trainedTargets[target] = { algorithm: "gp", gpByDim };
      continue;
    }
    if (cfg.type === "number") {
      const y = dataset.Y.map((row) => row[0]);
      const rf = new RandomForestRegression({ nEstimators: 80, maxFeatures: 0.8, seed: 42, replacement: true });
      rf.train(dataset.X, y);
      const yPred = rf.predict(dataset.X);
      const residuals = y.map((v, i) => v - yPred[i]);
      const residualSeries = new dfd.Series(residuals);
      trainedTargets[target] = {
        algorithm: "rf",
        rfModel: rf.toJSON(),
        residualVariance: Number(residualSeries.var()),
      };
      continue;
    }
    const classLabels = dataset.Y[0].map((_, idx) => idx);
    const yClass = dataset.Y.map((row) => row.indexOf(Math.max(...row)));
    const rf = new RandomForestClassifier({ nEstimators: 120, maxFeatures: 0.8, seed: 42, replacement: true });
    rf.train(dataset.X, yClass);
    trainedTargets[target] = {
      algorithm: "rf",
      rfModel: rf.toJSON(),
      classLabels,
    };
  }
  return {
    modelType: useRandomForestFallback ? "random-forest-fallback" : "gaussian-process-regression",
    backend: useRandomForestFallback ? "ml-random-forest" : "ml-matrix",
    hyperparams: { lengthScale, noiseVariance },
    metadata,
    encoding,
    targetConfigs,
    trainedTargets,
  };
}

// This function computes validation metrics for each user-visible column.
function validateModel(model, encodedRows, rawRows, splitMode) {
  const n = encodedRows.length;
  const idx = [...Array(n).keys()];
  const shuffledIdx = shuffled(idx);
  const folds = splitMode === "split"
    ? [{ train: shuffledIdx.slice(0, Math.max(1, Math.floor(0.8 * n))), test: shuffledIdx.slice(Math.max(1, Math.floor(0.8 * n))) }]
    : idx.map((i) => ({ train: idx.filter((j) => j !== i), test: [i] }));

  const perColumn = {};
  for (const c of model.userColumns) perColumn[c] = { yTrue: [], yPred: [], inside95: 0, count: 0 };

  for (const fold of folds) {
    const foldModel = trainInternalModel(
      fold.train.map((i) => encodedRows[i]),
      model.encoding,
      model.metadata,
      model.hyperparams.lengthScale
    );
    foldModel.userColumns = model.userColumns;

    for (const i of fold.test) {
      for (const c of model.userColumns) {
        const pred = predictTarget(foldModel, c, encodedRows[i]);
        perColumn[c].count += 1;
        if (pred.type === "number") {
          const yT = Number(rawRows[i][c]);
          perColumn[c].yTrue.push(yT);
          perColumn[c].yPred.push(pred.mu);
          if (yT >= pred.mu - 1.96 * pred.sd && yT <= pred.mu + 1.96 * pred.sd) perColumn[c].inside95 += 1;
        } else {
          const hit = String(rawRows[i][c]) === pred.label ? 1 : 0;
          perColumn[c].yTrue.push(1);
          perColumn[c].yPred.push(hit);
          if (hit === 1) perColumn[c].inside95 += 1;
        }
      }
    }
  }

  const report = {};
  for (const [c, b] of Object.entries(perColumn)) {
    const maeValue = mae(b.yTrue, b.yPred);
    const r2Value = r2(b.yTrue, b.yPred);
    const calibration = b.count ? (100 * b.inside95) / b.count : 0;
    let verdict = "Good predictive reliability.";
    if (r2Value < 0.5) verdict = "Weak predictive reliability. Improve data quality/volume.";
    else if (r2Value < 0.75) verdict = "Moderate predictive reliability.";
    if (model.metadata[c].originalType === "text") verdict = `${verdict} (Categorical metrics are hit/miss approximations.)`;
    report[c] = { type: model.metadata[c].originalType, mae: maeValue, r2: r2Value, calibration95Pct: calibration, verdict };
  }
  return report;
}

// This function prints the validation report in plain text.
function printValidationReport(report) {
  console.log("\nValidation Report:");
  for (const [column, metrics] of Object.entries(report)) {
    const warn = metrics.r2 < 0.5 ? " [WARNING: R2 < 0.5]" : "";
    console.log(`- ${column}: MAE=${metrics.mae.toFixed(4)}, R2=${metrics.r2.toFixed(4)}, 95% calibration=${metrics.calibration95Pct.toFixed(2)}%${warn}`);
    console.log(`  Verdict: ${metrics.verdict}`);
  }
}

// This function computes permutation feature importance for user-visible columns only.
function permutationImportance(model, encodedRows, rawRows) {
  const baselineByTarget = {};
  for (const target of model.userColumns) {
    const yTrue = [];
    const yPred = [];
    for (let i = 0; i < encodedRows.length; i += 1) {
      const pred = predictTarget(model, target, encodedRows[i]);
      if (pred.type === "number") {
        yTrue.push(Number(rawRows[i][target]));
        yPred.push(pred.mu);
      } else {
        yTrue.push(1);
        yPred.push(pred.label === String(rawRows[i][target]) ? 1 : 0);
      }
    }
    baselineByTarget[target] = mae(yTrue, yPred);
  }

  const importance = {};
  for (const sourceCol of model.userColumns) {
    const permutedRaw = shuffled(rawRows.map((r) => r[sourceCol]));
    const perturbed = encodedRows.map((row, idx) => {
      const copy = { ...row };
      if (model.encoding.numeric[sourceCol]) {
        const { min, max } = model.encoding.numeric[sourceCol];
        const val = Number(permutedRaw[idx]);
        copy[sourceCol] = max === min ? 0 : (val - min) / (max - min);
      } else {
        for (const cat of model.encoding.text[sourceCol].categories) copy[`${sourceCol}__${cat}`] = 0;
        const key = `${sourceCol}__${String(permutedRaw[idx])}`;
        if (key in copy) copy[key] = 1;
      }
      const internal = model.companionMap[sourceCol];
      if (internal) copy[internal] = 1;
      return copy;
    });

    let delta = 0;
    let count = 0;
    for (const target of model.userColumns) {
      if (target === sourceCol) continue;
      const yTrue = [];
      const yPred = [];
      for (let i = 0; i < perturbed.length; i += 1) {
        const pred = predictTarget(model, target, perturbed[i]);
        if (pred.type === "number") {
          yTrue.push(Number(rawRows[i][target]));
          yPred.push(pred.mu);
        } else {
          yTrue.push(1);
          yPred.push(pred.label === String(rawRows[i][target]) ? 1 : 0);
        }
      }
      delta += mae(yTrue, yPred) - baselineByTarget[target];
      count += 1;
    }
    importance[sourceCol] = count ? delta / count : 0;
  }

  return Object.entries(importance).sort((a, b) => b[1] - a[1]).map(([column, score], idx) => ({
    rank: idx + 1,
    column,
    importanceScore: score,
  }));
}

// This function computes Expected Improvement using Monte Carlo sampling.
function expectedImprovement(mu, sd, bestSoFar, nSamples = 200) {
  const sigma = Math.max(sd, 1e-9);
  let gain = 0;
  for (let i = 0; i < nSamples; i += 1) {
    const z = Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
    gain += Math.max(0, mu + sigma * z - bestSoFar);
  }
  return gain / nSamples;
}

// This function samples one random encoded candidate and sets internal flags safely.
function randomEncodedCandidate(model) {
  const row = {};
  for (const c of model.userColumns) {
    if (model.encoding.numeric[c]) row[c] = Math.random();
    else {
      const cats = model.encoding.text[c].categories;
      const pick = cats[Math.floor(Math.random() * cats.length)];
      for (const cat of cats) row[`${c}__${cat}`] = cat === pick ? 1 : 0;
    }
    const internal = model.companionMap[c];
    if (internal) row[internal] = 0;
  }
  return row;
}

// This function applies user-fixed values and missing companions to an encoded candidate.
function applyFixedValuesToEncoded(model, encoded, fixedValues, skippedColumns) {
  for (const c of model.userColumns) {
    const hasFixed = Object.prototype.hasOwnProperty.call(fixedValues, c);
    if (!hasFixed) continue;
    const v = fixedValues[c];
    if (model.encoding.numeric[c]) {
      const { min, max } = model.encoding.numeric[c];
      encoded[c] = max === min ? 0 : (Number(v) - min) / (max - min);
    } else {
      for (const cat of model.encoding.text[c].categories) encoded[`${c}__${cat}`] = 0;
      encoded[`${c}__${String(v)}`] = 1;
    }
  }
  for (const c of model.userColumns) {
    const companion = model.companionMap[c];
    if (!companion) continue;
    encoded[companion] = skippedColumns.has(c) ? 1 : 0;
  }
}

// This function ranks candidate points by expected improvement for a chosen target.
function rankCandidatesByEI(model, maximizeCol, candidates, observedBest) {
  const scored = candidates.map((encoded) => {
    const pred = predictTarget(model, maximizeCol, encoded);
    if (pred.type === "number") {
      const ei = expectedImprovement(pred.mu, pred.sd, observedBest);
      return { encoded, pred, objective: ei };
    }
    const bestClassProb = pred.ranked[0] ? pred.ranked[0].probability : 0;
    return { encoded, pred, objective: bestClassProb };
  });
  scored.sort((a, b) => b.objective - a.objective);
  return scored;
}

// This function stores suggestion sessions under the data directory.
function persistSuggestions(logs, dataDir) {
  if (!logs.length) return null;
  ensureDirectory(dataDir);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const file = path.resolve(dataDir, `suggestions_${stamp}.json`);
  fs.writeFileSync(file, JSON.stringify(logs, null, 2), "utf8");
  return file;
}

module.exports = {
  askQuestion,
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
  mean,
  decodeUserRow,
  randomEncodedCandidate,
  applyFixedValuesToEncoded,
  rankCandidatesByEI,
  persistSuggestions,
  predictTarget,
};
