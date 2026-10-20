//! BM25+ lexical scoring and convex-combination fusion.
//!
//! Pure arithmetic over term statistics — no storage, no tokenizer, no I/O — so
//! it stays wasm-portable like the rest of this crate, and is testable without
//! a database.
//!
//! # Why BM25+ rather than BM25
//!
//! Plain BM25's length normalization can drive a long document's contribution
//! for a matched term arbitrarily close to zero, so a long document that
//! genuinely contains the term can score below a short one that does not
//! contain it at all. BM25+ (Lv & Zhai, 2011) adds a constant `delta` to each
//! matched term's contribution, which bounds that below. In a vault where note
//! length varies by orders of magnitude that failure is routine, not exotic.
//!
//! # Why normalize against a theoretical maximum
//!
//! Fusing two channels requires their scores to be comparable. Normalizing each
//! against the best score *observed in the candidate set* (min-max) is the
//! obvious approach and is wrong: it stretches whatever the top hit scored to
//! 1.0, so a weak result set looks as strong as a confident one, and a channel
//! with an inherently compressed range is inflated to match one with a wide
//! range. Normalizing against the maximum each channel *could* attain for this
//! query keeps "weak" looking weak.

/// BM25+ tuning parameters.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Bm25Params {
    /// Term-frequency saturation. Higher means repeated terms keep adding.
    pub k1: f32,
    /// Length-normalization strength, in `0.0..=1.0`.
    pub b: f32,
    /// BM25+ lower bound on a matched term's contribution.
    pub delta: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        // k1/b are the long-standing BM25 defaults; delta = 0.5 is the value
        // Lv & Zhai report and that Seek uses.
        Self {
            k1: 1.2,
            b: 0.75,
            delta: 0.5,
        }
    }
}

/// Hybrid retrieval configuration.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Whether the lexical channel runs at all. With it off, ranking is
    /// bit-identical to pure dense retrieval — the fused path is not entered.
    pub lexical: bool,
    /// The convex-combination weight on the dense channel. 1.0 is dense-only,
    /// 0.0 lexical-only.
    pub dense_weight: f32,
    pub bm25: Bm25Params,
    /// Bounded additive bonus for a document whose *title* carries the query's
    /// terms, scaled by how much of the query the title covers.
    ///
    /// A note titled for a topic is usually what someone looking for that topic
    /// wants, rather than one of many notes mentioning it in passing.
    pub title_boost: f32,
    /// Largest bonus a brand-new note can earn for recency; 0 turns the signal
    /// off. The bonus halves every `recency_half_life_days`, and a note with no
    /// date earns none.
    pub recency_weight: f32,
    /// Age, in days, at which the recency bonus has halved.
    pub recency_half_life_days: f32,
    /// The moment ages are measured from, in Unix seconds. Set by the frontend
    /// rather than read from a clock here, so the pipeline stays usable in wasm
    /// and an evaluation can pin it. Recency is off while this is `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub now_unix: Option<i64>,
    /// Whether a query term the index has never seen is expanded to the
    /// vocabulary terms a typo away. A query that spells every term correctly
    /// is never touched.
    pub fuzzy: bool,
    /// How much a fuzzy expansion counts against an exact match, in
    /// `0.0..=1.0`. Below 1.0 so an exact match always outranks a fuzzy one.
    pub fuzzy_weight: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            lexical: true,
            // Measured, not chosen by taste. Sweeping alpha on two BEIR
            // datasets, 0.9 is the only value that improves *both* (nfcorpus
            // +0.008, scifact +0.014 nDCG@10). They disagree about the
            // optimum — nfcorpus peaks near 0.3, scifact near 0.9 — so a
            // single global weight is a compromise, and 0.9 is the compromise
            // that never regresses. See docs/gnosis/hybrid-lexical-fusion.md.
            dense_weight: 0.9,
            bm25: Bm25Params::default(),
            // Measured, and the measurement is shape-dependent. On known-item
            // lookups — searching for a note by its name, the query a notes
            // tool exists to answer — 0.1 lifts nDCG@10 from 0.38 to 0.50
            // against a ceiling of 0.57, i.e. from 66% to 87% of what is
            // achievable. On topical benchmark queries it costs about 0.006.
            // That asymmetry is roughly twentyfold in favour of enabling it,
            // so it is on by default and small.
            title_boost: 0.1,
            recency_weight: 0.0,
            recency_half_life_days: 365.0,
            now_unix: None,
            // Measured on kepano's vault with one typo per word of 5+
            // characters in short title queries: lexical-only nDCG@10 goes
            // 0.25 -> 0.80 and hybrid 0.32 -> 0.38, and scifact / nfcorpus
            // move by 0.001 and 0.002. The weight is flat from 0.3 to 0.7
            // (+-0.002), so 0.5 sits in the middle. The cost is a term that
            // is spelled right but absent from the index being matched to its
            // neighbors: lexical-only title queries lose about 0.01.
            fuzzy: true,
            fuzzy_weight: 0.5,
        }
    }
}

/// Corpus-level statistics the scorer needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorpusStats {
    /// Number of chunks in the searchable corpus.
    pub total_chunks: u64,
    /// Mean chunk length, in the same unit as [`LexicalMatch::chunk_len`].
    /// Must be > 0 for a non-empty corpus.
    pub avg_chunk_len: f32,
}

/// One query term's corpus statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryTerm {
    /// How many chunks contain this term.
    pub doc_freq: u64,
    /// How much a match counts: 1.0 for a term the query spelled, less for a
    /// fuzzy expansion of one it misspelled.
    pub weight: f32,
}

impl QueryTerm {
    /// A term the query spelled itself, at full weight.
    pub fn exact(doc_freq: u64) -> Self {
        Self {
            doc_freq,
            weight: 1.0,
        }
    }
}

/// What one candidate chunk contributes for a query.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalMatch {
    /// Chunk length, for length normalization — in **characters**, which is
    /// what every caller measures and what `avg_chunk_len` is averaged over.
    ///
    /// The unit is arbitrary as far as BM25+ is concerned, since `chunk_len`
    /// only ever appears divided by `avg_chunk_len`. What matters is that the
    /// two are the same unit: characters here against tokens in the average
    /// would scale the length-normalization term by the mean token length and
    /// quietly mistune `b`. These said "tokens" while SQLite's `LENGTH()` and
    /// the pipeline's `chars().count()` both counted characters — harmless as
    /// long as nothing believed the comment, which a second implementation
    /// would have.
    pub chunk_len: u32,
    /// Term frequency per query term, positionally aligned with the
    /// `&[QueryTerm]` passed alongside. Zero for a term this chunk lacks.
    pub term_freqs: Vec<u32>,
}

/// Split text into lexical terms.
///
/// Lowercases and splits on anything that is not alphanumeric, which
/// approximates SQLite FTS5's `unicode61` tokenizer closely enough that term
/// frequencies counted here line up with the candidates FTS5 matched. It is an
/// approximation, not a reimplementation: `unicode61` also folds diacritics by
/// default, which this does not, so an accented term can be matched by FTS5 and
/// then counted as absent here. The consequence is a conservative score for
/// that term rather than a wrong candidate set.
///
/// Scoring and tokenizing live together deliberately — a scorer whose notion of
/// a term differs from whatever produced its statistics is silently wrong.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Terms shorter than this are never expanded: one edit away from a three
/// letter word is half the dictionary.
pub const FUZZY_MIN_LEN: usize = 4;
/// Terms this long may be two edits off; shorter ones, one.
pub const FUZZY_LONG_LEN: usize = 8;
/// The most expansions one misspelled term gets, nearest first, so a query's
/// `OR` stays bounded however large the vocabulary.
pub const FUZZY_MAX_EXPANSIONS: usize = 8;

/// The edit distance within which a vocabulary term stands in for an unknown
/// `term`: none for a short one, 1 up to seven characters, 2 from eight.
pub fn fuzzy_bound(term: &str) -> usize {
    match term.chars().count() {
        n if n < FUZZY_MIN_LEN => 0,
        n if n < FUZZY_LONG_LEN => 1,
        _ => 2,
    }
}

/// Edit distance between `a` and `b` if it is at most `max`, else `None`.
///
/// Optimal string alignment: insertions, deletions, substitutions and adjacent
/// transpositions each cost one, since swapping two letters is the commonest
/// typo and plain Levenshtein would charge it two.
pub fn edit_distance_within(a: &str, b: &str, max: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    let width = b.len() + 1;
    let mut d = vec![0usize; (a.len() + 1) * width];
    for (j, cell) in d.iter_mut().take(width).enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        d[i * width] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (d[(i - 1) * width + j] + 1)
                .min(d[i * width + j - 1] + 1)
                .min(d[(i - 1) * width + j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(d[(i - 2) * width + j - 2] + 1);
            }
            d[i * width + j] = best;
        }
    }
    let got = d[a.len() * width + b.len()];
    (got <= max).then_some(got)
}

/// Collects, for terms the index has never seen, the vocabulary terms within
/// [`fuzzy_bound`] of them.
///
/// Fed one vocabulary term at a time, so each store can supply its vocabulary
/// however it holds it and both reach the same expansions: ordered nearest
/// first, then alphabetically, and capped at [`FUZZY_MAX_EXPANSIONS`].
pub struct FuzzyExpander<'a> {
    unknown: Vec<(usize, &'a str, usize)>,
    found: Vec<Vec<(usize, String)>>,
}

impl<'a> FuzzyExpander<'a> {
    /// `unknown` pairs each term with its position in the query.
    pub fn new(unknown: impl IntoIterator<Item = (usize, &'a str)>) -> Self {
        let unknown: Vec<_> = unknown
            .into_iter()
            .map(|(i, t)| (i, t, fuzzy_bound(t)))
            .filter(|&(_, _, bound)| bound > 0)
            .collect();
        let found = vec![Vec::new(); unknown.len()];
        Self { unknown, found }
    }

    /// Whether any term is eligible, so a caller can skip walking the
    /// vocabulary when none is.
    pub fn is_empty(&self) -> bool {
        self.unknown.is_empty()
    }

    pub fn offer(&mut self, word: &str) {
        for (&(_, term, bound), found) in self.unknown.iter().zip(&mut self.found) {
            if word == term {
                continue;
            }
            if let Some(distance) = edit_distance_within(term, word, bound) {
                found.push((distance, word.to_string()));
            }
        }
    }

    /// `(query position, expansion)` pairs.
    pub fn finish(self) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for ((index, _, _), mut found) in self.unknown.into_iter().zip(self.found) {
            found.sort();
            found.dedup();
            found.truncate(FUZZY_MAX_EXPANSIONS);
            out.extend(found.into_iter().map(|(_, word)| (index, word)));
        }
        out
    }
}

/// Count each of `terms` in `text`, returning counts positionally aligned with
/// `terms`. `terms` are expected to be already lowercased by `tokenize`.
pub fn term_freqs(text: &str, terms: &[String]) -> Vec<u32> {
    let mut counts = vec![0u32; terms.len()];
    for token in tokenize(text) {
        for (i, term) in terms.iter().enumerate() {
            if &token == term {
                counts[i] += 1;
            }
        }
    }
    counts
}

/// Share of the query's terms that appear in `title`, in `0.0..=1.0`.
///
/// A document whose *title* carries the query's words usually is the thing
/// being looked for, rather than a document that merely mentions them — the
/// note named "HNSW" versus twenty notes discussing it in passing.
///
/// Proportional rather than an all-or-nothing test for the title containing
/// the whole query. Full containment still scores 1.0 and so still gets the
/// largest boost, but a partial match is not discarded, which matters because
/// exact title matches are rare outside genuinely navigational queries. Terms
/// are compared after `tokenize`, so the match is case- and
/// punctuation-insensitive.
pub fn title_match(query_terms: &[String], title: &str) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let title_terms: std::collections::HashSet<String> = tokenize(title).into_iter().collect();
    let matched = query_terms
        .iter()
        .filter(|t| title_terms.contains(*t))
        .count();
    matched as f32 / query_terms.len() as f32
}

/// Inverse document frequency, in the form Lv & Zhai use for BM25+.
///
/// `ln((N + 1) / df)` is always positive, unlike the classic
/// `ln((N - df + 0.5) / (df + 0.5))`, which goes negative for a term appearing
/// in more than half the corpus. A negative IDF combined with BM25+'s `delta`
/// would let a very common term *subtract* from a score, which is not a
/// behaviour worth having.
fn idf(doc_freq: u64, total_chunks: u64) -> f32 {
    if doc_freq == 0 {
        return 0.0;
    }
    ((total_chunks as f32 + 1.0) / doc_freq as f32).ln()
}

/// BM25+ score for one chunk against a query.
///
/// `terms` and `m.term_freqs` are positionally aligned; a shorter
/// `term_freqs` is treated as zero for the missing terms.
pub fn score(
    m: &LexicalMatch,
    terms: &[QueryTerm],
    stats: CorpusStats,
    params: Bm25Params,
) -> f32 {
    if terms.is_empty() || stats.total_chunks == 0 || stats.avg_chunk_len <= 0.0 {
        return 0.0;
    }
    let norm_len = params.k1
        * (1.0 - params.b + params.b * (m.chunk_len as f32 / stats.avg_chunk_len));

    terms
        .iter()
        .enumerate()
        .map(|(i, term)| {
            let tf = m.term_freqs.get(i).copied().unwrap_or(0) as f32;
            if tf <= 0.0 {
                // A term this chunk lacks contributes nothing — `delta` is a
                // floor on *matched* terms, not a reward for absent ones.
                return 0.0;
            }
            let saturated = ((params.k1 + 1.0) * tf) / (norm_len + tf);
            term.weight * idf(term.doc_freq, stats.total_chunks) * (saturated + params.delta)
        })
        .sum()
}

/// The largest BM25+ score this query could attain, for normalization.
///
/// As `tf` grows the saturating term approaches `k1 + 1`, so each query term is
/// bounded by `idf * (k1 + 1 + delta)`. Summing those gives a per-query ceiling
/// that depends only on the query and the corpus — never on what happened to be
/// retrieved — which is what makes the normalized score comparable across
/// queries and across channels.
pub fn max_score(terms: &[QueryTerm], stats: CorpusStats, params: Bm25Params) -> f32 {
    terms
        .iter()
        .map(|t| t.weight * idf(t.doc_freq, stats.total_chunks) * (params.k1 + 1.0 + params.delta))
        .sum()
}

/// [`max_score`] for a query some of whose terms are fuzzy expansions.
///
/// `groups` gives, for each term, the index of the query term it stands for.
/// Expansions of one misspelled term are alternatives for the same word, so the
/// ceiling counts the best of them once rather than summing them; a query with
/// no expansions has every term in its own group and gets [`max_score`] back.
pub fn max_score_grouped(
    terms: &[QueryTerm],
    groups: &[usize],
    stats: CorpusStats,
    params: Bm25Params,
) -> f32 {
    let mut best: std::collections::BTreeMap<usize, f32> = Default::default();
    for (t, &group) in terms.iter().zip(groups) {
        let ceiling = t.weight * idf(t.doc_freq, stats.total_chunks) * (params.k1 + 1.0 + params.delta);
        let slot = best.entry(group).or_insert(0.0);
        *slot = slot.max(ceiling);
    }
    best.values().sum()
}

/// Scale a raw score into `0.0..=1.0` against a ceiling, clamping.
///
/// A zero or negative ceiling yields 0.0 rather than a division by zero: that
/// is the "no query term appears anywhere in the corpus" case, where no lexical
/// evidence exists and the channel should contribute nothing.
pub fn normalize(raw: f32, ceiling: f32) -> f32 {
    if ceiling <= 0.0 {
        return 0.0;
    }
    (raw / ceiling).clamp(0.0, 1.0)
}

/// Combine a normalized dense score and a normalized lexical score.
///
/// `dense_weight` is the `alpha` of the convex combination: 1.0 is dense-only
/// (today's behaviour exactly), 0.0 is lexical-only. Values outside
/// `0.0..=1.0` are clamped rather than rejected, so a mistyped config degrades
/// to an endpoint instead of failing a search.
pub fn fuse(dense_norm: f32, lexical_norm: f32, dense_weight: f32) -> f32 {
    let alpha = dense_weight.clamp(0.0, 1.0);
    alpha * dense_norm + (1.0 - alpha) * lexical_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(total: u64, avg: f32) -> CorpusStats {
        CorpusStats {
            total_chunks: total,
            avg_chunk_len: avg,
        }
    }

    fn m(len: u32, tfs: &[u32]) -> LexicalMatch {
        LexicalMatch {
            chunk_len: len,
            term_freqs: tfs.to_vec(),
        }
    }

    fn terms(dfs: &[u64]) -> Vec<QueryTerm> {
        dfs.iter().map(|&doc_freq| QueryTerm::exact(doc_freq)).collect()
    }

    // ---- idf --------------------------------------------------------------

    #[test]
    fn idf_is_higher_for_rarer_terms() {
        let rare = idf(1, 1000);
        let common = idf(500, 1000);
        assert!(rare > common, "{rare} !> {common}");
    }

    #[test]
    fn idf_stays_positive_even_for_a_term_in_every_chunk() {
        // The classic IDF formula goes negative past df > N/2; this one must
        // not, or a common term would subtract from a score via delta.
        assert!(idf(1000, 1000) > 0.0);
    }

    #[test]
    fn idf_of_an_absent_term_is_zero() {
        assert_eq!(idf(0, 1000), 0.0);
    }

    // ---- score ------------------------------------------------------------

    #[test]
    fn a_chunk_containing_the_term_outscores_one_that_does_not() {
        let t = terms(&[5]);
        let s = stats(100, 50.0);
        let has = score(&m(50, &[3]), &t, s, Bm25Params::default());
        let lacks = score(&m(50, &[0]), &t, s, Bm25Params::default());
        assert!(has > lacks, "{has} !> {lacks}");
        assert_eq!(lacks, 0.0);
    }

    #[test]
    fn more_occurrences_score_higher_but_with_diminishing_returns() {
        let t = terms(&[5]);
        let s = stats(100, 50.0);
        let p = Bm25Params::default();
        let at = |tf| score(&m(50, &[tf]), &t, s, p);

        assert!(at(2) > at(1));
        assert!(at(10) > at(2));

        // Saturation is about the *marginal* gain of one more occurrence, so
        // compare equal-sized increments at different points on the curve.
        // (Comparing 1->2 against 2->10 would compare one increment with
        // eight, which proves nothing.)
        let early = at(2) - at(1);
        let late = at(10) - at(9);
        assert!(
            early > late,
            "each further occurrence must add less: {early} !> {late}"
        );
    }

    #[test]
    fn a_rarer_term_contributes_more() {
        let s = stats(1000, 50.0);
        let p = Bm25Params::default();
        let rare = score(&m(50, &[2]), &terms(&[2]), s, p);
        let common = score(&m(50, &[2]), &terms(&[800]), s, p);
        assert!(rare > common, "{rare} !> {common}");
    }

    #[test]
    fn a_longer_chunk_scores_lower_for_the_same_term_count() {
        let t = terms(&[5]);
        let s = stats(100, 50.0);
        let p = Bm25Params::default();
        let short = score(&m(10, &[2]), &t, s, p);
        let long = score(&m(500, &[2]), &t, s, p);
        assert!(short > long, "{short} !> {long}");
    }

    /// The reason BM25+ exists: a long chunk that genuinely contains the term
    /// must still beat one that does not contain it at all, however long.
    #[test]
    fn delta_keeps_a_long_match_above_a_non_match() {
        let t = terms(&[5]);
        let s = stats(100, 50.0);
        let p = Bm25Params::default();
        let very_long_match = score(&m(100_000, &[1]), &t, s, p);
        let short_non_match = score(&m(5, &[0]), &t, s, p);
        assert!(
            very_long_match > short_non_match,
            "BM25+ must floor a match above a non-match: {very_long_match} !> {short_non_match}"
        );
        assert!(very_long_match > 0.0);
    }

    #[test]
    fn delta_zero_reduces_to_plain_bm25() {
        let t = terms(&[5]);
        let s = stats(100, 50.0);
        let plain = Bm25Params {
            delta: 0.0,
            ..Bm25Params::default()
        };
        // With delta 0 a sufficiently long document's contribution tends to 0.
        let very_long = score(&m(10_000_000, &[1]), &t, s, plain);
        assert!(very_long < 1e-3, "expected ~0 without delta, got {very_long}");
    }

    #[test]
    fn multiple_query_terms_sum() {
        let s = stats(100, 50.0);
        let p = Bm25Params::default();
        let one_term = score(&m(50, &[2, 0]), &terms(&[5, 5]), s, p);
        let both_terms = score(&m(50, &[2, 2]), &terms(&[5, 5]), s, p);
        assert!(both_terms > one_term);
    }

    #[test]
    fn missing_term_freq_entries_count_as_zero() {
        let s = stats(100, 50.0);
        let p = Bm25Params::default();
        let explicit = score(&m(50, &[2, 0]), &terms(&[5, 5]), s, p);
        let truncated = score(&m(50, &[2]), &terms(&[5, 5]), s, p);
        assert_eq!(explicit, truncated);
    }

    #[test]
    fn b_zero_disables_length_normalization() {
        let t = terms(&[5]);
        let s = stats(100, 50.0);
        let p = Bm25Params {
            b: 0.0,
            ..Bm25Params::default()
        };
        let short = score(&m(10, &[2]), &t, s, p);
        let long = score(&m(5000, &[2]), &t, s, p);
        assert!((short - long).abs() < 1e-6, "length must not matter: {short} vs {long}");
    }

    // ---- degenerate inputs ------------------------------------------------

    #[test]
    fn an_empty_query_scores_zero() {
        assert_eq!(score(&m(50, &[]), &[], stats(100, 50.0), Bm25Params::default()), 0.0);
    }

    #[test]
    fn an_empty_corpus_scores_zero() {
        let p = Bm25Params::default();
        assert_eq!(score(&m(50, &[2]), &terms(&[1]), stats(0, 0.0), p), 0.0);
    }

    #[test]
    fn a_zero_length_chunk_does_not_divide_by_zero() {
        let got = score(&m(0, &[2]), &terms(&[5]), stats(100, 50.0), Bm25Params::default());
        assert!(got.is_finite(), "got {got}");
    }

    // ---- max_score and normalize ------------------------------------------

    #[test]
    fn no_score_can_exceed_the_theoretical_maximum() {
        let t = terms(&[3, 7]);
        let s = stats(500, 40.0);
        let p = Bm25Params::default();
        let ceiling = max_score(&t, s, p);
        // A chunk stuffed with both terms, as short as possible.
        let extreme = score(&m(1, &[100_000, 100_000]), &t, s, p);
        assert!(
            extreme <= ceiling + 1e-4,
            "score {extreme} exceeded ceiling {ceiling}"
        );
    }

    #[test]
    fn the_maximum_depends_only_on_the_query_and_corpus() {
        let s = stats(500, 40.0);
        let p = Bm25Params::default();
        // Same query, different retrieved sets -> same ceiling. This is the
        // property min-max normalization lacks.
        assert_eq!(max_score(&terms(&[3, 7]), s, p), max_score(&terms(&[3, 7]), s, p));
        assert!(max_score(&terms(&[3]), s, p) < max_score(&terms(&[3, 7]), s, p));
    }

    #[test]
    fn normalize_maps_into_the_unit_range() {
        assert_eq!(normalize(0.0, 10.0), 0.0);
        assert_eq!(normalize(5.0, 10.0), 0.5);
        assert_eq!(normalize(10.0, 10.0), 1.0);
    }

    #[test]
    fn normalize_clamps_rather_than_overshooting() {
        assert_eq!(normalize(25.0, 10.0), 1.0);
        assert_eq!(normalize(-3.0, 10.0), 0.0);
    }

    #[test]
    fn normalize_against_a_zero_ceiling_is_zero_not_nan() {
        let got = normalize(1.0, 0.0);
        assert!(got.is_finite() && got == 0.0, "got {got}");
    }

    /// A weak result set must stay weak. Min-max normalization would stretch
    /// whatever topped the list to 1.0; a theoretical ceiling does not.
    #[test]
    fn a_weak_best_hit_does_not_normalize_to_one() {
        let ceiling = 20.0;
        assert!(normalize(1.5, ceiling) < 0.1);
    }

    // ---- fuse -------------------------------------------------------------

    #[test]
    fn dense_weight_one_is_dense_only() {
        assert_eq!(fuse(0.8, 0.1, 1.0), 0.8);
    }

    #[test]
    fn dense_weight_zero_is_lexical_only() {
        assert_eq!(fuse(0.8, 0.1, 0.0), 0.1);
    }

    #[test]
    fn fuse_is_a_convex_combination() {
        let fused = fuse(1.0, 0.0, 0.7);
        assert!((fused - 0.7).abs() < 1e-6, "got {fused}");
    }

    #[test]
    fn fuse_is_monotonic_in_each_channel() {
        assert!(fuse(0.9, 0.5, 0.5) > fuse(0.4, 0.5, 0.5));
        assert!(fuse(0.5, 0.9, 0.5) > fuse(0.5, 0.4, 0.5));
    }

    #[test]
    fn fuse_clamps_an_out_of_range_weight() {
        assert_eq!(fuse(0.8, 0.1, 2.0), 0.8, "above 1 behaves as dense-only");
        assert_eq!(fuse(0.8, 0.1, -1.0), 0.1, "below 0 behaves as lexical-only");
    }

    #[test]
    fn fuse_of_two_unit_scores_stays_in_range() {
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let got = fuse(1.0, 1.0, alpha);
            assert!((got - 1.0).abs() < 1e-6, "alpha {alpha} gave {got}");
        }
    }

    // ---- tokenize / term_freqs -------------------------------------------

    #[test]
    fn tokenize_lowercases_and_splits_on_punctuation() {
        assert_eq!(tokenize("HNSW graph-construction, fast!"), vec!["hnsw", "graph", "construction", "fast"]);
    }

    #[test]
    fn tokenize_keeps_digits_and_alphanumeric_runs() {
        assert_eq!(tokenize("bge-small-en-v1.5"), vec!["bge", "small", "en", "v1", "5"]);
    }

    #[test]
    fn tokenize_discards_empty_runs() {
        assert_eq!(tokenize("  ...  a   b  "), vec!["a", "b"]);
    }

    #[test]
    fn tokenize_of_nothing_is_empty() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("---").is_empty());
    }

    #[test]
    fn term_freqs_counts_each_term_positionally() {
        let terms = vec!["graph".to_string(), "missing".to_string(), "hnsw".to_string()];
        let got = term_freqs("HNSW graph: a graph of graphs", &terms);
        assert_eq!(got, vec![2, 0, 1], "'graphs' must not count as 'graph'");
    }

    #[test]
    fn term_freqs_is_case_insensitive() {
        let terms = vec!["hnsw".to_string()];
        assert_eq!(term_freqs("HNSW hnsw HnSw", &terms), vec![3]);
    }

    #[test]
    fn term_freqs_of_an_empty_query_is_empty() {
        assert!(term_freqs("anything at all", &[]).is_empty());
    }

    /// A term counted here must be scored the same way the corpus statistics
    /// were gathered, so a round trip through tokenize is the contract.
    #[test]
    fn term_freqs_agrees_with_tokenize() {
        let text = "Vector search, vector indexes; VECTOR!";
        let terms = tokenize("vector");
        assert_eq!(term_freqs(text, &terms), vec![3]);
    }

    // ---- title_match -----------------------------------------------------

    fn terms_of(q: &str) -> Vec<String> {
        tokenize(q)
    }

    #[test]
    fn a_title_containing_the_whole_query_scores_one() {
        assert_eq!(title_match(&terms_of("hnsw graph"), "HNSW graph construction"), 1.0);
    }

    #[test]
    fn a_title_containing_none_of_the_query_scores_zero() {
        assert_eq!(title_match(&terms_of("hnsw graph"), "Chocolate cake recipe"), 0.0);
    }

    #[test]
    fn a_partial_title_match_scores_proportionally() {
        // One of two terms present.
        assert!((title_match(&terms_of("hnsw cake"), "HNSW graph") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn title_match_is_case_and_punctuation_insensitive() {
        assert_eq!(title_match(&terms_of("hnsw"), "HNSW!"), 1.0);
        assert_eq!(title_match(&terms_of("graph construction"), "Graph-Construction"), 1.0);
    }

    #[test]
    fn title_match_counts_each_query_term_once() {
        // A title repeating a term must not score above 1.0.
        assert_eq!(title_match(&terms_of("graph"), "graph graph graph"), 1.0);
    }

    #[test]
    fn title_match_of_an_empty_query_is_zero() {
        assert_eq!(title_match(&[], "anything"), 0.0);
    }

    #[test]
    fn title_match_of_an_empty_title_is_zero() {
        assert_eq!(title_match(&terms_of("hnsw"), ""), 0.0);
    }

    /// The failure mode worth watching: a short common query matching many
    /// titles boosts them all equally, which flattens rather than orders. The
    /// scoring cannot prevent that on its own — it is why the boost is bounded
    /// and additive rather than dominant.
    #[test]
    fn a_common_single_term_query_boosts_every_matching_title_equally() {
        let q = terms_of("notes");
        assert_eq!(title_match(&q, "Meeting notes"), 1.0);
        assert_eq!(title_match(&q, "Reading notes"), 1.0);
    }

    // ---- fuzzy matching ----------------------------------------------------

    #[test]
    fn edit_distance_counts_each_kind_of_edit_once() {
        assert_eq!(edit_distance_within("graph", "graph", 2), Some(0));
        assert_eq!(edit_distance_within("graph", "graps", 2), Some(1), "substitution");
        assert_eq!(edit_distance_within("graph", "grap", 2), Some(1), "deletion");
        assert_eq!(edit_distance_within("graph", "graphs", 2), Some(1), "insertion");
        assert_eq!(edit_distance_within("graph", "garph", 2), Some(1), "transposition");
    }

    #[test]
    fn edit_distance_beyond_the_bound_is_none() {
        assert_eq!(edit_distance_within("graph", "grxpy", 1), None);
        assert_eq!(edit_distance_within("graph", "grxpy", 2), Some(2));
        assert_eq!(edit_distance_within("graph", "gr", 2), None, "length alone rules it out");
    }

    #[test]
    fn edit_distance_counts_characters_not_bytes() {
        assert_eq!(edit_distance_within("café", "cafe", 1), Some(1));
    }

    #[test]
    fn the_distance_bound_grows_with_the_term() {
        assert_eq!(fuzzy_bound("cat"), 0, "too short to expand");
        assert_eq!(fuzzy_bound("cake"), 1);
        assert_eq!(fuzzy_bound("graphs"), 1);
        assert_eq!(fuzzy_bound("retrieval"), 2, "eight characters or more");
        assert_eq!(fuzzy_bound("naïveté!"), 2, "counted in characters");
    }

    fn expand(unknown: &[&str], vocabulary: &[&str]) -> Vec<(usize, String)> {
        let mut e = FuzzyExpander::new(unknown.iter().copied().enumerate());
        for w in vocabulary {
            e.offer(w);
        }
        e.finish()
    }

    #[test]
    fn an_unknown_term_expands_to_the_vocabulary_within_its_bound() {
        let got = expand(&["grph"], &["graph", "group", "graphs", "cake"]);
        assert_eq!(got, vec![(0, "graph".to_string())], "graphs is two edits, bound is one");
    }

    #[test]
    fn a_longer_term_reaches_two_edits() {
        let got = expand(&["retreival"], &["retrieval", "retrieve"]);
        assert_eq!(got, vec![(0, "retrieval".to_string())], "a transposition, one edit");
        let got = expand(&["retrievl"], &["retrieval"]);
        assert_eq!(got.len(), 1);
        let got = expand(&["retrievxx"], &["retrieval"]);
        assert_eq!(got.len(), 1, "two edits is allowed from eight characters");
    }

    #[test]
    fn a_short_term_is_never_expanded() {
        assert!(expand(&["cat"], &["cut", "car", "cats"]).is_empty());
    }

    #[test]
    fn expansions_are_nearest_first_then_alphabetical_and_capped() {
        // Twenty words two edits from the term, and one a single edit away
        // that sorts after all of them.
        let far: Vec<String> = (0..20).map(|i| format!("stormcloudx{}", (b'a' + i) as char)).collect();
        let mut vocab: Vec<&str> = far.iter().map(String::as_str).collect();
        vocab.push("stormclouds");
        let got = expand(&["stormcloud"], &vocab);
        assert_eq!(got.len(), FUZZY_MAX_EXPANSIONS);
        assert_eq!(got[0].1, "stormclouds", "the nearest comes first");
        assert_eq!(got[1].1, "stormcloudxa", "then alphabetically");
    }

    #[test]
    fn an_expansion_never_repeats_the_term_itself() {
        assert!(expand(&["graph"], &["graph"]).is_empty());
    }

    #[test]
    fn a_fuzzy_term_counts_at_its_weight() {
        let s = stats(100, 50.0);
        let p = Bm25Params::default();
        let at = |weight| {
            let t = [QueryTerm { doc_freq: 5, weight }];
            score(&m(50, &[3]), &t, s, p)
        };
        assert!((at(0.5) - 0.5 * at(1.0)).abs() < 1e-6);
        assert_eq!(at(1.0), score(&m(50, &[3]), &terms(&[5]), s, p));
    }

    #[test]
    fn an_exact_match_outscores_the_same_match_found_fuzzily() {
        let s = stats(100, 50.0);
        let p = Bm25Params::default();
        let exact = score(&m(50, &[2]), &[QueryTerm::exact(5)], s, p);
        let fuzzy = score(&m(50, &[2]), &[QueryTerm { doc_freq: 5, weight: 0.7 }], s, p);
        assert!(exact > fuzzy, "{exact} !> {fuzzy}");
    }

    #[test]
    fn grouped_ceiling_without_expansions_is_the_plain_ceiling() {
        let s = stats(500, 40.0);
        let p = Bm25Params::default();
        let t = terms(&[3, 7]);
        assert_eq!(max_score_grouped(&t, &[0, 1], s, p), max_score(&t, s, p));
    }

    #[test]
    fn expansions_of_one_term_share_a_ceiling() {
        let s = stats(500, 40.0);
        let p = Bm25Params::default();
        let spelled = QueryTerm::exact(0);
        let a = QueryTerm { doc_freq: 3, weight: 0.5 };
        let b = QueryTerm { doc_freq: 9, weight: 0.5 };
        let grouped = max_score_grouped(&[spelled, a, b], &[0, 0, 0], s, p);
        let best = max_score_grouped(&[a], &[0], s, p);
        assert_eq!(grouped, best, "alternatives for one word count once, at the best");
    }
}
