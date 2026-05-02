// This function exits with a clear error and command-specific help text.
function failWithHelp(command, message) {
  console.error(`Error: ${message}`);
  if (command && typeof command.outputHelp === "function") {
    console.error("");
    command.outputHelp();
  }
  process.exit(1);
}

module.exports = { failWithHelp };
