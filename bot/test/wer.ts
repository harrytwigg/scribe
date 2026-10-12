const words = (text: string) => text.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(Boolean)

/** Word error rate: edit distance over words, divided by the reference length. */
export function wer(reference: string, hypothesis: string): number {
  const ref = words(reference)
  const hyp = words(hypothesis)
  let row = Array.from({ length: hyp.length + 1 }, (_, j) => j)
  for (let i = 1; i <= ref.length; i++) {
    const next = [i]
    for (let j = 1; j <= hyp.length; j++) {
      next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1))
    }
    row = next
  }
  return row[hyp.length] / Math.max(ref.length, 1)
}
