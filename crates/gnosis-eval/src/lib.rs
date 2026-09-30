//! Relevance metrics for scoring gnosis retrieval quality.
//!
//! Pure arithmetic over ranked path lists and graded judgements — no native
//! dependencies, no I/O, no knowledge of how results were retrieved. Loading a
//! qrels file and running real searches lives in the CLI; this crate only
//! answers "how good was that ranking".
//!
//! # Grades
//!
//! A judgement carries a graded relevance: `2` = ideal, `1` = relevant, `0` =
//! explicitly judged irrelevant. A document with no judgement at all is scored
//! as irrelevant too, but [`unjudged_at`] counts those separately — a rising
//! unjudged count means the annotations are going stale rather than that
//! quality improved.

/// One human judgement: a vault-relative document path and its grade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Judged {
    pub path: String,
    pub grade: u8,
}

impl Judged {
    pub fn new(path: impl Into<String>, grade: u8) -> Self {
        Self {
            path: path.into(),
            grade,
        }
    }
}

/// Grade of `path`, or 0 when it carries no judgement.
fn grade_of(path: &str, judged: &[Judged]) -> u8 {
    judged
        .iter()
        .find(|j| j.path == path)
        .map(|j| j.grade)
        .unwrap_or(0)
}

/// Normalized discounted cumulative gain over the top `k` results.
///
/// Uses the exponential gain `2^grade - 1` with a `log2(rank + 1)` discount,
/// normalized against the best ranking the judgements allow. Returns `0.0`
/// when no judged document is relevant, since there is no attainable gain to
/// normalize against.
pub fn ndcg_at(ranked: &[String], judged: &[Judged], k: usize) -> f64 {
    let dcg = discounted_gain(ranked.iter().take(k).map(|p| grade_of(p, judged)));

    // The best attainable ranking: the highest grades first, capped at the
    // same cutoff, so retrieving the single best document at k=1 scores 1.0
    // rather than being punished for documents k had no room for.
    let mut ideal: Vec<u8> = judged.iter().map(|j| j.grade).filter(|g| *g > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg = discounted_gain(ideal.into_iter().take(k));

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Sum of `(2^grade - 1) / log2(rank + 1)` over a ranked run of grades.
fn discounted_gain(grades: impl Iterator<Item = u8>) -> f64 {
    grades
        .enumerate()
        .map(|(i, grade)| {
            let gain = 2f64.powi(i32::from(grade)) - 1.0;
            gain / ((i as f64) + 2.0).log2()
        })
        .sum()
}

/// Fraction of all relevant documents (grade >= 1) that appear in the top `k`.
///
/// Returns `0.0` when nothing is relevant — the metric is undefined there, and
/// 0 keeps a query with no attainable score from inflating an average.
pub fn recall_at(ranked: &[String], judged: &[Judged], k: usize) -> f64 {
    let total_relevant = judged.iter().filter(|j| j.grade > 0).count();
    if total_relevant == 0 {
        return 0.0;
    }
    let found = ranked
        .iter()
        .take(k)
        .filter(|p| grade_of(p, judged) > 0)
        .count();
    found as f64 / total_relevant as f64
}

/// Reciprocal of the 1-based rank of the first relevant result, or `0.0` if
/// no relevant document was retrieved. Unbounded in `k` by definition.
pub fn mrr(ranked: &[String], judged: &[Judged]) -> f64 {
    ranked
        .iter()
        .position(|p| grade_of(p, judged) > 0)
        .map(|i| 1.0 / ((i as f64) + 1.0))
        .unwrap_or(0.0)
}

/// How many of the top `k` results carry no judgement at all.
///
/// Distinct from "scored zero": an unjudged hit may well be relevant and
/// simply unannotated, so this is a measure of qrels coverage, not quality.
pub fn unjudged_at(ranked: &[String], judged: &[Judged], k: usize) -> usize {
    ranked
        .iter()
        .take(k)
        .filter(|p| !judged.iter().any(|j| &&j.path == p))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // ---- nDCG ------------------------------------------------------------

    #[test]
    fn ndcg_is_one_for_the_ideal_ranking() {
        let ranked = paths(&["a.md", "b.md", "c.md"]);
        let judged = vec![Judged::new("a.md", 2), Judged::new("b.md", 1)];
        assert!(close(ndcg_at(&ranked, &judged, 10), 1.0));
    }

    #[test]
    fn ndcg_penalizes_a_worse_ordering() {
        let judged = vec![Judged::new("a.md", 2), Judged::new("b.md", 1)];
        let ideal = ndcg_at(&paths(&["a.md", "b.md"]), &judged, 10);
        let swapped = ndcg_at(&paths(&["b.md", "a.md"]), &judged, 10);
        assert!(
            swapped < ideal,
            "ranking the grade-1 doc above the grade-2 doc must score lower: {swapped} !< {ideal}"
        );
    }

    #[test]
    fn ndcg_matches_a_hand_computed_value() {
        // Ranked: b(grade 1), a(grade 2).
        //   DCG  = (2^1-1)/log2(2) + (2^2-1)/log2(3) = 1.0 + 3/1.5849625007 = 2.8927892607
        //   IDCG = (2^2-1)/log2(2) + (2^1-1)/log2(3) = 3.0 + 1/1.5849625007 = 3.6309297536
        //   nDCG = 0.7966004...
        let ranked = paths(&["b.md", "a.md"]);
        let judged = vec![Judged::new("a.md", 2), Judged::new("b.md", 1)];
        let expected = (1.0 + 3.0 / 3f64.log2()) / (3.0 + 1.0 / 3f64.log2());
        assert!(
            close(ndcg_at(&ranked, &judged, 10), expected),
            "got {}, want {expected}",
            ndcg_at(&ranked, &judged, 10)
        );
    }

    #[test]
    fn ndcg_is_zero_when_nothing_relevant_was_judged() {
        let ranked = paths(&["a.md", "b.md"]);
        // Grade 0 is an explicit "not relevant", so there is no gain to reach.
        let judged = vec![Judged::new("a.md", 0)];
        assert!(close(ndcg_at(&ranked, &judged, 10), 0.0));
    }

    #[test]
    fn ndcg_is_zero_when_no_relevant_document_was_retrieved() {
        let ranked = paths(&["x.md", "y.md"]);
        let judged = vec![Judged::new("a.md", 2)];
        assert!(close(ndcg_at(&ranked, &judged, 10), 0.0));
    }

    #[test]
    fn ndcg_respects_the_cutoff() {
        // The only relevant doc sits at rank 3, outside k=2.
        let ranked = paths(&["x.md", "y.md", "a.md"]);
        let judged = vec![Judged::new("a.md", 2)];
        assert!(close(ndcg_at(&ranked, &judged, 2), 0.0));
        assert!(ndcg_at(&ranked, &judged, 3) > 0.0);
    }

    #[test]
    fn ndcg_ideal_ranking_is_capped_at_k() {
        // Three relevant docs but k=1: retrieving the best one is a perfect
        // score at that cutoff, not one third of one.
        let ranked = paths(&["a.md"]);
        let judged = vec![
            Judged::new("a.md", 2),
            Judged::new("b.md", 2),
            Judged::new("c.md", 2),
        ];
        assert!(close(ndcg_at(&ranked, &judged, 1), 1.0));
    }

    #[test]
    fn ndcg_handles_k_larger_than_the_result_set() {
        let ranked = paths(&["a.md"]);
        let judged = vec![Judged::new("a.md", 2)];
        assert!(close(ndcg_at(&ranked, &judged, 100), 1.0));
    }

    #[test]
    fn ndcg_of_an_empty_ranking_is_zero() {
        let judged = vec![Judged::new("a.md", 2)];
        assert!(close(ndcg_at(&[], &judged, 10), 0.0));
    }

    // ---- recall ----------------------------------------------------------

    #[test]
    fn recall_counts_relevant_documents_found() {
        let ranked = paths(&["a.md", "x.md", "b.md"]);
        let judged = vec![
            Judged::new("a.md", 2),
            Judged::new("b.md", 1),
            Judged::new("c.md", 1),
        ];
        assert!(close(recall_at(&ranked, &judged, 10), 2.0 / 3.0));
    }

    #[test]
    fn recall_ignores_documents_judged_irrelevant() {
        let ranked = paths(&["a.md", "z.md"]);
        let judged = vec![Judged::new("a.md", 1), Judged::new("z.md", 0)];
        assert!(
            close(recall_at(&ranked, &judged, 10), 1.0),
            "a grade-0 judgement is not a relevant document to be found"
        );
    }

    #[test]
    fn recall_respects_the_cutoff() {
        let ranked = paths(&["x.md", "a.md"]);
        let judged = vec![Judged::new("a.md", 2)];
        assert!(close(recall_at(&ranked, &judged, 1), 0.0));
        assert!(close(recall_at(&ranked, &judged, 2), 1.0));
    }

    #[test]
    fn recall_is_zero_when_nothing_is_relevant() {
        let ranked = paths(&["a.md"]);
        let judged = vec![Judged::new("a.md", 0)];
        assert!(close(recall_at(&ranked, &judged, 10), 0.0));
    }

    // ---- MRR -------------------------------------------------------------

    #[test]
    fn mrr_is_the_reciprocal_of_the_first_relevant_rank() {
        let ranked = paths(&["x.md", "y.md", "a.md"]);
        let judged = vec![Judged::new("a.md", 1)];
        assert!(close(mrr(&ranked, &judged), 1.0 / 3.0));
    }

    #[test]
    fn mrr_is_one_when_the_top_hit_is_relevant() {
        let ranked = paths(&["a.md", "x.md"]);
        let judged = vec![Judged::new("a.md", 2)];
        assert!(close(mrr(&ranked, &judged), 1.0));
    }

    #[test]
    fn mrr_ignores_grade_zero_judgements() {
        let ranked = paths(&["z.md", "a.md"]);
        let judged = vec![Judged::new("z.md", 0), Judged::new("a.md", 1)];
        assert!(close(mrr(&ranked, &judged), 1.0 / 2.0));
    }

    #[test]
    fn mrr_is_zero_when_no_relevant_document_was_retrieved() {
        let ranked = paths(&["x.md"]);
        let judged = vec![Judged::new("a.md", 2)];
        assert!(close(mrr(&ranked, &judged), 0.0));
    }

    // ---- unjudged --------------------------------------------------------

    #[test]
    fn unjudged_counts_results_with_no_annotation() {
        let ranked = paths(&["a.md", "x.md", "y.md"]);
        let judged = vec![Judged::new("a.md", 2)];
        assert_eq!(unjudged_at(&ranked, &judged, 10), 2);
    }

    #[test]
    fn unjudged_does_not_count_an_explicit_irrelevant_judgement() {
        let ranked = paths(&["z.md", "x.md"]);
        let judged = vec![Judged::new("z.md", 0)];
        assert_eq!(
            unjudged_at(&ranked, &judged, 10),
            1,
            "grade 0 is an annotation; only x.md is unjudged"
        );
    }

    #[test]
    fn unjudged_respects_the_cutoff() {
        let ranked = paths(&["x.md", "y.md", "z.md"]);
        assert_eq!(unjudged_at(&ranked, &[], 2), 2);
    }
}
