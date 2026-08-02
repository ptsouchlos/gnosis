import "infra/just/shells.just"
import "infra/rust/just/clippy.just"
import "infra/rust/just/format.just"

default:
    @just -l

[doc('Build binary target in debug for native CPU')]
[group('dev')]
build-target target:
    cargo rustc -p {{ target }} --bin {{ target }} -- -C target-cpu=native

[doc('Build binary target in debug for native CPU')]
[group('dev')]
build-target-release target:
    cargo rustc --release -p {{ target }} --bin {{ target }} -- -C target-cpu=native

[doc('Build the project (default is debug)')]
[group('dev')]
build config="debug":
    echo "Building the project..."
    cargo build --workspace --all-targets {{ if config == "release" { "--release" } else { "" } }}

[doc('Build and run tests (default is debug)')]
[group('dev')]
test config="debug":
    echo "Running tests..."
    cargo test --workspace --all-targets {{ if config == "release" { "--release" } else { "" } }}
    cargo test --workspace --doc

[doc('Build and run all tests, including ignored ones')]
test-all config="debug":
    cargo test --workspace --all-targets {{ if config == "release" { "--release" } else { "" } }} -- --include-ignored
    cargo test --workspace --doc
