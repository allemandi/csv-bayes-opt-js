#!/usr/bin/env node
const path = require("path");
const { Command } = require("commander");
const { failWithHelp } = require("./utils/cli-utils");
const { runTrain } = require("./commands/train");
const { runValidate } = require("./commands/validate");
const { runSuggest } = require("./commands/suggest");

// This function configures command routing and delegates work to command modules.
async function main() {
  const defaultDataDir = path.resolve(process.cwd(), "data");
  const program = new Command();

  program
    .name("csv-bayes-opt")
    .description("CSV relationship learner with train/validate/suggest workflows.")
    .showHelpAfterError()
    .configureOutput({ outputError: (str, write) => write(str) });

  program
    .command("train")
    .description("Train a model from CSV and save JSON model.")
    .option("--csv <path>", "Path to training CSV file", path.join(defaultDataDir, "samples.csv"))
    .option("--output <path>", "Path to output model JSON", path.join(defaultDataDir, "model.json"))
    .action(async (opts, cmd) => {
      try {
        await runTrain(opts.csv, opts.output);
      } catch (err) {
        failWithHelp(cmd, err.message || String(err));
      }
    });

  program
    .command("validate")
    .description("Load and print stored validation report from model JSON.")
    .option("--model <path>", "Path to model JSON file", path.join(defaultDataDir, "model.json"))
    .action((opts, cmd) => {
      try {
        runValidate(opts.model);
      } catch (err) {
        failWithHelp(cmd, err.message || String(err));
      }
    });

  program
    .command("suggest")
    .description("Load model and produce optimization suggestions.")
    .option("--model <path>", "Path to model JSON file", path.join(defaultDataDir, "model.json"))
    .option("--maximize <column>", "Target column to maximize (numeric or categorical; if omitted, prompt interactively)")
    .action(async (opts, cmd) => {
      try {
        await runSuggest(opts.model, opts.maximize, defaultDataDir);
      } catch (err) {
        failWithHelp(cmd, err.message || String(err));
      }
    });

  if (!process.argv.slice(2).length) {
    program.outputHelp();
    process.exit(0);
  }
  try {
    await program.parseAsync(process.argv);
  } catch (err) {
    failWithHelp(program, err.message || String(err));
  }
}

main();
