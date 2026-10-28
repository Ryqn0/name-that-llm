// "Do you know who wrote it?": the visitor's optional answer after a result (P6).
//
// Visitors who know which chatbot wrote what they pasted turn the public site into a live
// test set: per model, how often the champion (and the challenger, in shadow mode) named
// the right author. That only works if each answer is counted ONCE, against the result it
// was given for. Each result therefore keeps its own labels and accepts one answer, even
// when the visitor answers an older result after a newer one appeared.
// A pure module (no DOM): the page wires it, and tests/test_feedback.py runs it in Node.

export const OTHER_AUTHORS = ["another model", "a person"]; // beyond the 18 models
let lastId = 0;

// One result the visitor may answer: the labels it showed (or computed, for the
// challenger), and whether it has been answered.
export function newResult({ label, challenger_label = null, band }) {
  lastId += 1;
  return { id: lastId, label, challenger_label, band, answered: false };
}

// The `feedback` event's fields for the visitor's choice `actual`, or null: for a result
// already answered, or a choice that's neither one of `labels` (the model's 18) nor one of
// OTHER_AUTHORS. A refused choice doesn't use up the result's one answer.
export function answer(result, actual, labels) {
  if (result.answered) return null;
  if (!labels.includes(actual) && !OTHER_AUTHORS.includes(actual)) return null;
  result.answered = true;
  return {
    actual,
    label: result.label,
    challenger_label: result.challenger_label,
    band: result.band,
  };
}
