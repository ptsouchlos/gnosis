//! Progress-reporting interface for gnosis's indexing pipeline. Kept free of
//! any concrete backend's dependencies (e.g. `indicatif`) so it can be
//! depended on by non-terminal consumers (e.g. a future server reporting
//! progress over a websocket) that need only the shape of progress
//! reporting, not a terminal progress bar.

/// Reports progress for a long-running operation with a growing total.
pub trait Progress {
    /// Increase the known total unit of work (e.g. as more files are
    /// discovered — the total isn't known upfront).
    fn inc_total(&self, delta: u64);

    /// Advance progress by `delta` units.
    fn inc(&self, delta: u64);

    /// Mark the operation complete.
    fn finish(&self);
}

/// No-op [`Progress`] for callers that don't want a bar.
pub struct NoopProgress;

impl Progress for NoopProgress {
    fn inc_total(&self, _delta: u64) {}
    fn inc(&self, _delta: u64) {}
    fn finish(&self) {}
}
