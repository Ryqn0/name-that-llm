// Anonymous telemetry for the public demo (P6): what the page reports, and nothing more.
//
// The page sends small events to Grafana Cloud's collector, so a dashboard can show visits,
// speed, errors and how the models do. The pasted text never leaves the browser, in any
// form (no excerpt, no hash, no word list), and that's built in rather than promised:
//   - ALLOWLIST: an event keeps only the fields its spec (SPECS below) names, each checked
//     against its type: a finite number, one of the model's labels, a valid choice, a
//     short string. Anything else is dropped before it could be sent, a stray "text"
//     field included.
//   - REDACTION: a free-text field (an error's kind or message) holding a run of 4 or more
//     letters that also occurs in the visitor's text becomes "[redacted]". An error
//     message that happens to quote the passage, even cut mid-word, can't carry it out.
//   - NO IDENTIFIERS: no cookie, no storage, and no session or user id. A payload's meta
//     names the app and its version only, so no two visits can be linked.
// send() is fire-and-forget and never throws: monitoring can't break or slow the page.

const OTHER_AUTHORS = ["another model", "a person"]; // "It was actually..." beyond the 18
const UNUSUAL = ["short", "unknown_words", "non_latin"];
const LETTERS = /\p{L}{4,}/gu; // runs of 4+ letters, in any script

// Each event's fields, and the type each must have. A string type is checked by check();
// an array is the list of valid choices; {max} is free text cut to max characters.
const SPECS = {
  page_load: { load_ms: "number", ok: "boolean", what: ["champion", "challenger"] },
  answer: {
    answer_ms: "number",
    words: "words",
    label: "label",
    band: "band",
    challenger_label: "label or null",
    unusual: "unusual",
  },
  feedback: { actual: "actual", label: "label", challenger_label: "label or null", band: "band" },
  error: { where: ["load", "answer", "feedback"], kind: { max: 40 }, message: { max: 120 } },
};

// A word count in steps of 25: enough to see the texts' lengths, too coarse to single
// one visit out.
export function wordBucket(n) {
  return 25 * Math.floor(n / 25);
}

// One event, {name, fields}, keeping only its allowlisted, well-typed fields.
// ctx is {labels, bands, text}: the model's labels, its "How sure?" bands, and the
// visitor's text, which is used only to redact and is never part of the result.
export function makeEvent(name, fields, ctx) {
  const spec = SPECS[name];
  if (!spec) throw new Error(`unknown event: ${name}`);
  const given = fields ?? {};
  const out = {};
  for (const [key, type] of Object.entries(spec)) {
    if (!Object.hasOwn(given, key)) continue;
    const value = check(type, given[key], ctx);
    if (value !== undefined) out[key] = value; // undefined: the value failed its check
  }
  return { name, fields: out };
}

function check(type, value, ctx) {
  const isLabel = (v) => (ctx.labels ?? []).includes(v);
  if (type === "number") return Number.isFinite(value) ? Math.round(value) : undefined;
  if (type === "words") return Number.isFinite(value) && value >= 0 ? wordBucket(value) : undefined;
  if (type === "boolean") return typeof value === "boolean" ? value : undefined;
  if (type === "label") return isLabel(value) ? value : undefined;
  if (type === "label or null") return value === null || isLabel(value) ? value : undefined;
  if (type === "actual") return isLabel(value) || OTHER_AUTHORS.includes(value) ? value : undefined;
  if (type === "band") return (ctx.bands ?? []).some((b) => b.label === value) ? value : undefined;
  if (type === "unusual") {
    const ok = Array.isArray(value) && value.every((v) => UNUSUAL.includes(v));
    return ok ? [...new Set(value)] : undefined;
  }
  if (Array.isArray(type)) return type.includes(value) ? value : undefined;
  if (typeof value !== "string") return undefined;
  const short = value.slice(0, type.max);
  return leaks(short, ctx.text ?? "") ? "[redacted]" : short;
}

// True if `value` holds a run of 4+ letters that also occurs anywhere in `text`. Runs are
// looked up as substrings of the text, so a word cut at the field's maximum length, or a
// fragment of a script written without spaces, is still caught.
function leaks(value, text) {
  const haystack = text.toLowerCase();
  return (value.toLowerCase().match(LETTERS) ?? []).some((run) => haystack.includes(run));
}

// The collector's JSON (Grafana Faro): each event with string attributes, which are its
// fields plus the context needed to compare models (model_run, challenger_run, synthetic).
// meta = {commit, model_run, challenger_run, synthetic}.
export function toCollectorPayload(events, meta) {
  const context = {
    model_run: meta.model_run,
    challenger_run: meta.challenger_run,
    synthetic: meta.synthetic, // true only for the robot visitor (P6 Task 6)
  };
  const timestamp = new Date().toISOString();
  return {
    events: events.map(({ name, fields }) => ({
      name,
      domain: "demo",
      attributes: asStrings({ ...fields, ...context }),
      timestamp,
    })),
    // The app's name and version only: no session, user, page or browser part.
    meta: {
      app: {
        name: "name-that-llm-demo",
        version: String(meta.commit ?? ""),
        environment: meta.synthetic ? "synthetic" : "production",
      },
      // The collector refuses a payload that names no SDK ("no SDK version"). This isn't
      // Grafana's Faro SDK, so it says so: this module, by name and version.
      sdk: { name: "name-that-llm-telemetry", version: "1.0.0" },
    },
  };
}

// The collector takes string attributes only: lists become "a,b", and missing values
// (null, undefined) are left out.
function asStrings(record) {
  const out = {};
  for (const [key, value] of Object.entries(record)) {
    if (value === null || value === undefined) continue;
    out[key] = Array.isArray(value) ? value.join(",") : String(value);
  }
  return out;
}

// POST the payload to the collector. Fire-and-forget: the page never waits on it, and the
// promise never rejects. Without a url (no collector configured), nothing is sent.
export async function send(payload, url, fetchImpl = globalThis.fetch) {
  if (!url) return;
  try {
    await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Faro-Session-Id": randomId() },
      body: JSON.stringify(payload),
      keepalive: true, // the request outlives the page if the visitor leaves
    });
  } catch {
    // A failed send is simply lost: monitoring must never break the page.
  }
}

// A fresh random id for every request. Grafana's collector refuses a request without an
// X-Faro-Session-Id header (its own SDK tracks sessions). A new one each time satisfies it
// while linking nothing: not two visits, not even two events of the same visit.
function randomId() {
  const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
