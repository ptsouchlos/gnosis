// Does a malformed PDF trap the extraction — and only the extraction?
//
// `parse_pdf` is unwind-safe in a native build only because it wraps
// `pdf-extract` in `catch_unwind`, so the panic it raises is caught before it
// can reach the `panic = "abort"` handler. The `wasm-release` profile uses that
// strategy, where `catch_unwind` cannot catch a panic — so a malformed content
// stream does not return an error, it *traps* the call. The plugin (step 3)
// runs each file's extraction in its own disposable instance for exactly this
// reason, so a bad PDF is a handled per-file boundary rather than a crash.
// This test pins that boundary against the built package.
//
//   node tests/node/trap.mjs <pkg-dir> <fixture-dir>
//
// Both paths come in as arguments; nothing is read from this script's own
// directory, and the artifact is read as data only.

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [pkgDir, fixtureDir] = process.argv.slice(2);
if (!pkgDir || !fixtureDir) {
  console.error("usage: node trap.mjs <pkg-dir> <fixture-dir>");
  process.exit(2);
}

// The `web` target fetches its `.wasm` by URL, which does not work for a local
// file in node — so the bytes go in explicitly. This is the same module the
// plugin loads, rather than a node-specific build of it.
const mod = await import(pathToFileURL(resolve(pkgDir, "gnosis_extract.js")).href);
const wasmBytes = await readFile(resolve(pkgDir, "gnosis_extract_bg.wasm"));

// The `web` target keeps a single instance behind the module, so re-initializing
// it with the same bytes is how the plugin's "a fresh instance per file" is
// modeled here: a disposable worker cannot share a module across files.
const freshInstance = () => mod.default({ module_or_path: wasmBytes });

const sample = new Uint8Array(await readFile(resolve(fixtureDir, "sample.pdf")));
const malformed = new Uint8Array(await readFile(resolve(fixtureDir, "malformed-ops.pdf")));

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok   " : "FAIL "} ${name}${detail ? " — " + detail : ""}`);
  if (!ok) failures += 1;
};

// 1. A valid PDF extracts: the bundle is actually wired up before we trust it
//    to fail cleanly.
await freshInstance();
const extracted = mod.extractPdf(sample);
check(
  "a valid PDF extracts its title and pages",
  extracted.title === "Gnosis Sample Document" &&
    extracted.pages.length === 2 &&
    extracted.pages[0].page === 1 &&
    extracted.pages[0].text.trim().length > 0,
);

// 2. A malformed content stream traps the call. A throw here is the expected
//    outcome, not a defect: the trap is what the per-file worker relies on to
//    contain the failure.
await freshInstance();
let trapped = false;
let trapName = "";
try {
  mod.extractPdf(malformed);
} catch (e) {
  trapped = true;
  trapName = e.constructor.name;
}
check("a malformed content stream traps the call", trapped, trapped ? trapName : "");

// 3. The trap is confined to that instance: the process is alive and a fresh
//    instance still extracts a valid PDF. This is the guarantee step 3 builds
//    on — one bad PDF costs only itself.
await freshInstance();
let recovered = false;
try {
  const after = mod.extractPdf(sample);
  recovered = after.title === "Gnosis Sample Document" && after.pages.length === 2;
} catch {
  recovered = false;
}
check("a fresh instance still extracts a valid PDF after the trap", recovered);

if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("a malformed PDF traps the extraction, and only the extraction");
