deny_checks := env('DENY_CHECKS', 'licenses bans sources')

[doc('Check dependency licenses, bans and sources with cargo-deny (DENY_CHECKS picks the checks; add advisories to include them)')]
[group('deps')]
deny *args:
    cargo deny check {{deny_checks}} {{args}}

[doc('Find unused dependencies with cargo-machete')]
[group('deps')]
machete *args:
    cargo machete {{args}}
