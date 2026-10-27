// A Node harness for tests/test_browser.py and the deploy's smoke test: it runs the same
// engine as the page (ntl.js), reading a JSON request on stdin and answering on stdout.
//   node cli.mjs flatten   {"texts": [...]}                 -> {"texts": [...]}
//   node cli.mjs predict   {"model": dir, "texts": [...]}   -> {"probabilities": [[18], ...]}
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { flatten, loadModel, predictProba } from "./ntl.js";

const command = process.argv[2];
const request = JSON.parse(readFileSync(0, "utf8"));

if (command === "flatten") {
  answer({ texts: request.texts.map(flatten) });
} else if (command === "predict") {
  const json = JSON.parse(readFileSync(join(request.model, "model.json"), "utf8"));
  const bytes = readFileSync(join(request.model, "weights.bin"));
  // A fresh, aligned ArrayBuffer: Float32Array needs a byte offset that's a multiple of 4.
  const weights = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const model = loadModel(json, weights);
  answer({ probabilities: request.texts.map((text) => predictProba(model, text)) });
} else {
  console.error(`unknown command: ${command} (expected flatten or predict)`);
  process.exit(2);
}

function answer(body) {
  process.stdout.write(JSON.stringify(body));
}
