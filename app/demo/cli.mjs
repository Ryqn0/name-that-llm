// A Node harness for tests/test_browser.py and the deploy's smoke test: it runs the same
// engine as the page (ntl.js), reading a JSON request on stdin and answering on stdout.
//   node cli.mjs flatten   {"texts": [...]}                 -> {"texts": [...]}
//   node cli.mjs predict   {"model": dir, "texts": [...]}   -> {"probabilities": [[18], ...]}
//   node cli.mjs stats     {"model": dir, "texts": [...]}   -> {"stats": [...], "reasons": [...]}
// And for tests/test_telemetry.py, the page's telemetry (telemetry.js):
//   node cli.mjs event     {"events": [{"name", "fields"}], "ctx": {...}, "meta": {...}}
//                          -> {"events": [...], "payload": {...}}, or {"error": "..."}
//   node cli.mjs send      {"url": "..."}                   -> {"resolved": true}
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { flatten, loadModel, predictProba, textStats, unusualReasons } from "./ntl.js";
import { makeEvent, send, toCollectorPayload } from "./telemetry.js";

const command = process.argv[2];
const request = JSON.parse(readFileSync(0, "utf8"));

if (command === "flatten") {
  answer({ texts: request.texts.map(flatten) });
} else if (command === "predict") {
  const model = readModel(request.model);
  answer({ probabilities: request.texts.map((text) => predictProba(model, text)) });
} else if (command === "stats") {
  const model = readModel(request.model);
  answer({
    stats: request.texts.map((text) => textStats(model, text)),
    reasons: request.texts.map((text) => unusualReasons(model, text)),
  });
} else if (command === "event") {
  try {
    const events = request.events.map((e) => makeEvent(e.name, e.fields, request.ctx));
    answer({ events, payload: toCollectorPayload(events, request.meta) });
  } catch (error) {
    answer({ error: error.message });
  }
} else if (command === "send") {
  await send({ events: [] }, request.url); // never rejects, or this would crash
  answer({ resolved: true });
} else {
  console.error(`unknown command: ${command} (expected flatten, predict, stats, event or send)`);
  process.exit(2);
}

function readModel(dir) {
  const json = JSON.parse(readFileSync(join(dir, "model.json"), "utf8"));
  const bytes = readFileSync(join(dir, "weights.bin"));
  // A fresh, aligned ArrayBuffer: Float32Array needs a byte offset that's a multiple of 4.
  const weights = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return loadModel(json, weights);
}

function answer(body) {
  process.stdout.write(JSON.stringify(body));
}
