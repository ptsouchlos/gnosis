//! Each submodule implements one `gnosis` subcommand: it defines the command's
//! `*Args` struct (the CLI input) and an `execute` function that runs it.

pub mod forget;
pub mod index;
pub mod init;
pub mod related;
pub mod rebuild;
pub mod search;
pub mod status;

/// Print an indexing run's collected errors (files skipped rather than
/// aborting the run — currently only possible for images), shared by
/// `index`/`rebuild`. No-op when there are none.
pub(crate) fn print_index_errors(errors: &[::index::IndexError]) {
    if errors.is_empty() {
        return;
    }
    println!("{} file(s) failed to index:", errors.len());
    for err in errors {
        println!("  {} — {}", err.path, err.message);
    }
}
