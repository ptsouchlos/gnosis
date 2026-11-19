profile := env('CARGO_PROFILE', 'dev')
bin := env('CARGO_BIN', '')
bin_flag := if bin != '' { '--bin ' + bin } else { '' }

[doc('Build the crate (CARGO_PROFILE selects the profile, default dev; CARGO_BIN a single binary)')]
[group('build')]
build *args:
    cargo build --profile {{profile}} {{bin_flag}} {{args}}

[doc('Build the crate with the release profile (CARGO_BIN selects a single binary)')]
[group('build')]
build-release *args:
    cargo build --release {{bin_flag}} {{args}}

[doc('Run a binary (CARGO_BIN selects it; pass cargo args before --, program args after)')]
[group('build')]
run *args:
    cargo run --profile {{profile}} {{bin_flag}} {{args}}

[doc('Run a binary with the release profile (CARGO_BIN selects it)')]
[group('build')]
run-release *args:
    cargo run --release {{bin_flag}} {{args}}

[doc('Run the tests')]
[group('test')]
test *args:
    cargo test {{args}}

[doc('Type-check without producing binaries')]
[group('build')]
check *args:
    cargo check --all-targets {{args}}

[doc('List the binaries of every workspace package as "package<TAB>binary" (needs jq)')]
[group('info')]
list-bins:
    cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | .name as $p | .targets[] | select(.kind | index("bin")) | "\($p)\t\(.name)"'

[doc('List every target of every workspace package as "package<TAB>target<TAB>kinds" (needs jq)')]
[group('info')]
list-targets:
    cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | .name as $p | .targets[] | "\($p)\t\(.name)\t\(.kind | join(","))"'

[doc('Remove the target directory')]
[group('build')]
clean:
    cargo clean
