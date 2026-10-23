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

[doc('Benchmark indexing throughput on a synthetic vault (file-count, image-count)')]
[group('dev')]
bench file_count="2000" image_count="200":
    infra/bench/run.sh {{ file_count }} {{ image_count }}

[doc('Build the wasm query client (wasm-pack + wasm-opt) and report its size')]
[group('wasm')]
wasm-build out="":
    script/wasm/build.sh {{ out }}

[doc('Write the third-party notices for the crates the wasm query client links')]
[group('wasm')]
wasm-notices out="target/THIRD_PARTY_NOTICES.txt":
    script/release/notices.sh {{ out }}

[doc('Run the wasm bindings tests in node')]
[group('wasm')]
wasm-test:
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
        cargo test -p gnosis-wasm --target wasm32-unknown-unknown
