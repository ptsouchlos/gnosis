cov_format := env('COV_FORMAT', '')
_cov_flag := if cov_format == 'lcov' { '--lcov --output-path lcov.info' } else if cov_format != '' { '--' + cov_format } else { '' }
crap_lcov := env('CRAP_LCOV', 'lcov.info')

[doc('Run Rust tests with coverage via cargo-llvm-cov (COV_FORMAT=lcov|html|summary-only)')]
[group('test')]
coverage *args:
    cargo llvm-cov {{_cov_flag}} {{args}}

[doc('Compute CRAP scores via cargo-crap (runs cargo-llvm-cov first; set CRAP_LCOV to change output path)')]
[group('test')]
crap *args:
    cargo llvm-cov --lcov --output-path {{crap_lcov}}
    cargo crap --lcov {{crap_lcov}} {{args}}

[doc('Clean cargo-llvm-cov artifacts')]
[group('test')]
coverage-clean:
    cargo llvm-cov clean --workspace
