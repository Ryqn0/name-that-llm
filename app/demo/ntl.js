// The browser demo's engine (P4 revision): the TF-IDF model `lr`, run in the visitor's
// browser. A port of exactly what Python computes:
//   flatten()       name_that_llm.data.flatten_text (chatbot markdown -> one paragraph)
//   predictProba()  scikit-learn's TfidfVectorizer (words + characters) and the
//                   multinomial LogisticRegression, from the files that
//                   name_that_llm.browser.export_browser_model writes
// tests/test_browser.py runs this file with Node and checks it against Python number for
// number. No DOM here: the same module serves the page and the tests.

// ---------- flatten: chatbot markdown -> one plain paragraph (see data.flatten_text) ----------
const FENCE = /^(```|~~~)/u;
const TABLE_RULE = /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$/u;
const QUOTE = /^(>\s?)+/u;
// \p{Nd}, not \d: Python's \d also matches non-ASCII digits, JavaScript's doesn't.
const LIST_OR_HEADING = /^(#{1,6}|[-*+]|\p{Nd}+[.)])\s+/u;
const IMAGE = /!\[([^\]]*)\]\([^)]*\)/gu;
const LINK = /\[([^\]]+)\]\([^)]*\)/gu;
// Python's str.splitlines() breaks on all of these.
const LINE_BREAKS = /\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/u;

export function flatten(text) {
  const lines = [];
  for (let line of text.split(LINE_BREAKS)) {
    line = line.trim();
    if (FENCE.test(line) || TABLE_RULE.test(line)) continue;
    line = line.replace(QUOTE, "").replace(LIST_OR_HEADING, "");
    lines.push(line);
  }
  let flat = lines.join(" ").replace(IMAGE, "$1").replace(LINK, "$1");
  flat = flat.replaceAll("**", "").replaceAll("__", "").replaceAll("`", "").replaceAll("|", " ");
  return flat.replace(/\s+/gu, " ").normalize("NFC").trim();
}

// ---------- the model ----------
export function loadModel(json, weights) {
  const f = new Float32Array(weights);
  const nWords = json.words.vocabulary.length;
  const nChars = json.chars.vocabulary.length;
  const n = nWords + nChars;
  const k = json.labels.length;
  let at = 0;
  const take = (size) => f.subarray(at, (at += size));
  const words = { ...json.words, idf: take(nWords), offset: 0 };
  const chars = { ...json.chars, idf: take(nChars), offset: nWords };
  const coef = take(k * n);
  const intercept = take(k);
  for (const block of [words, chars]) {
    block.index = new Map(block.vocabulary.map((term, i) => [term, i]));
  }
  // The single words the model knows (its vocabulary minus the bigrams): textStats' basis.
  const unigrams = new Set(words.vocabulary.filter((term) => !term.includes(" ")));
  return { ...json, words, chars, n, k, coef, intercept, unigrams };
}

// Python's (?u)\b\w\w+\b: maximal runs of word characters (letters, digits, underscore),
// at least 2 code points long. (JavaScript's own \w and \b are ASCII-only.)
const WORD_RUN = /[\p{L}\p{N}_]+/gu;

function ngrams(units, [lo, hi], join) {
  const out = [];
  for (let n = lo; n <= Math.min(hi, units.length); n++) {
    for (let i = 0; i + n <= units.length; i++) out.push(units.slice(i, i + n).join(join));
  }
  return out;
}

function wordTerms(text, block) {
  const doc = block.lowercase ? text.toLowerCase() : text;
  const tokens = (doc.match(WORD_RUN) || []).filter((t) => [...t].length >= 2);
  return ngrams(tokens, block.ngram_range, " ");
}

function charTerms(text, block) {
  let doc = block.lowercase ? text.toLowerCase() : text;
  doc = doc.replace(/\s\s+/gu, " "); // scikit-learn's char analyzer squeezes runs of whitespace
  return ngrams([...doc], block.ngram_range, ""); // code points, like Python's str
}

// One TF-IDF block: 1 + ln(count), times idf, then l2-normalized. Sparse: [column, value].
function tfidf(terms, block) {
  const counts = new Map();
  for (const term of terms) {
    const j = block.index.get(term);
    if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
  }
  const entries = [];
  let squares = 0;
  for (const [j, count] of counts) {
    const value = (1 + Math.log(count)) * block.idf[j];
    entries.push([block.offset + j, value]);
    squares += value * value;
  }
  const norm = Math.sqrt(squares);
  return norm > 0 ? entries.map(([j, value]) => [j, value / norm]) : [];
}

// The 18 probabilities, in model.labels order, for one passage (flattened first).
export function predictProba(model, text) {
  const flat = flatten(text);
  const x = [
    ...tfidf(wordTerms(flat, model.words), model.words),
    ...tfidf(charTerms(flat, model.chars), model.chars),
  ];
  const z = Array.from(model.intercept);
  for (let c = 0; c < model.k; c++) {
    const row = c * model.n;
    for (const [j, value] of x) z[c] += model.coef[row + j] * value;
  }
  const top = Math.max(...z);
  const exp = z.map((v) => Math.exp(v - top)); // softmax, shifted for numerical safety
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

// ---------- for the page ----------
export function rank(model, probabilities) {
  return model.labels
    .map((label, i) => ({ model: label, lab: model.labs[label], probability: probabilities[i] }))
    .sort((a, b) => b.probability - a.probability);
}

export function describeConfidence(top, bands) {
  return bands.find((band) => top >= band.min_top);
}

export function countWords(text) {
  const flat = flatten(text);
  return flat ? flat.split(" ").length : 0;
}

// ---------- unusual texts (P6, I-025) ----------
// A text unlike the training passages gets a note on the page, since the model's answer
// deserves less trust there. The limits were measured on our own passages
// (name_that_llm.browser.unusual_thresholds) and are shipped in model.meta.unusual.
const LETTER = /\p{L}/u;
const LATIN = /\p{Script=Latin}/u;

// {words, unknown_share, non_latin_share} for one text, flattened first like predictProba.
// unknown_share is the share of its single words outside the model's vocabulary
// (browser.unknown_share in Python computes the same number).
export function textStats(model, text) {
  const flat = flatten(text);
  const tokens = wordTerms(flat, { ...model.words, ngram_range: [1, 1] });
  const unknown = tokens.filter((token) => !model.unigrams.has(token)).length;
  const letters = [...flat].filter((c) => LETTER.test(c));
  const latin = letters.filter((c) => LATIN.test(c)).length;
  return {
    words: flat ? flat.split(" ").length : 0,
    unknown_share: tokens.length ? unknown / tokens.length : 0,
    non_latin_share: letters.length ? (letters.length - latin) / letters.length : 0,
  };
}

// Why a text looks unusual: any of "short", "unknown_words" and "non_latin", or none.
// Without limits in the model's meta (an older export), nothing is flagged.
export function unusualReasons(model, text) {
  const limits = model.meta?.unusual;
  if (!limits) return [];
  const stats = textStats(model, text);
  const reasons = [];
  if (stats.words < limits.min_words) reasons.push("short");
  if (stats.unknown_share > limits.max_unknown_share) reasons.push("unknown_words");
  if (stats.non_latin_share > limits.max_non_latin_share) reasons.push("non_latin");
  return reasons;
}
