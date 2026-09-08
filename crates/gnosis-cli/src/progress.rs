//! `indicatif`-backed [`Progress`] implementation. Native (depends on
//! `indicatif`), so it lives in the CLI binary crate rather than the
//! `progress` interface crate — mirrors how `store.rs`'s `SqliteStore` and
//! `walk.rs`'s `FsWalker` stay out of their interface crates.
use indicatif::{ProgressBar, ProgressStyle};
pub use progress::Progress;

/// `indicatif`-backed [`Progress`]: a determinate bar showing position/total.
pub struct IndicatifProgress(ProgressBar);

impl IndicatifProgress {
    pub fn new() -> Self {
        let bar = ProgressBar::new(0);
        bar.set_style(
            ProgressStyle::with_template("[{bar:40.cyan/blue}] {pos}/{len}")
                .expect("valid progress bar template"),
        );
        Self(bar)
    }
}

impl Default for IndicatifProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl Progress for IndicatifProgress {
    fn inc_total(&self, delta: u64) {
        self.0.inc_length(delta);
    }

    fn inc(&self, delta: u64) {
        self.0.inc(delta);
    }

    fn finish(&self) {
        self.0.finish_and_clear();
    }
}
