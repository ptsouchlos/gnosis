# gnosis

`gnosis` is a CLI tool for creating embeddings and performing semantic search
on text, image and PDF data using only local resources and models. You own your
data so it never leaves your machine. `gnosis` allows you to gain insights
from your knowledge base without relying on a 3rd party API or service.

## Releasing

A release publishes the wasm query client that the
[Obsidian plugin](https://github.com/ptsouchlos/obsidian-gnosis) vendors.

1. Bump `version` in the root `Cargo.toml`'s `[workspace.package]`, which every
   crate takes, and merge it.
2. Tag that commit with the same version and push the tag:

   ```sh
   git tag v0.2.0 origin/main
   git push origin v0.2.0
   ```

`.github/workflows/release.yml` refuses a tag that does not match the version.
It builds the bundle and attaches `gnosis-wasm-<tag>.tar.gz` and its `.sha256`
to a GitHub release, then asks the plugin to re-vendor it, which opens a pull
request there.

---

## Author

| [<img src="https://avatars0.githubusercontent.com/u/6591180?s=460&v=4" width="100"><br><sub>@ptsouchlos</sub>](https://github.com/ptsouchlos) |
| :-------------------------------------------------------------------------------------------------------------------------------------------: |
