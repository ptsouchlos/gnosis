[doc('Format all Rust code')]
[group('dev')]
format:
    cargo fmt --all

[doc('Check formatting without modifying files')]
[group('dev')]
format-check:
    cargo fmt --all --check
