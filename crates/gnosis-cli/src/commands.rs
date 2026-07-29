//! Each submodule implements one `gnosis` subcommand: it defines the command's
//! `*Args` struct (the CLI input) and an `execute` function that runs it.

pub mod forget;
pub mod index;
pub mod init;
pub mod related;
pub mod rebuild;
pub mod search;
pub mod status;
