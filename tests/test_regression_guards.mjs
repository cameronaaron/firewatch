import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const workerSource = readFileSync(path.join(repoRoot, "app/static/_worker.js"), "utf8");
const appSource = readFileSync(path.join(repoRoot, "app/static/app.js"), "utf8");
const indexSource = readFileSync(path.join(repoRoot, "app/static/index.html"), "utf8");

test("worker impact assessment logic is not duplicated", () => {
  const impactFnCount = (workerSource.match(/function buildImpactAssessment\s*\(/g) || []).length;
  const nearbyElevatedCount = (workerSource.match(/const nearbyElevated\s*=/g) || []).length;

  assert.equal(impactFnCount, 1, "buildImpactAssessment should be defined once");
  assert.equal(nearbyElevatedCount, 1, "nearbyElevated guard should be declared once");
});

test("impact driver weights remain normalized for 24h and 72h horizons", () => {
  const match = workerSource.match(/const driverInputs = \[(.*?)\n  \];/s);
  assert.ok(match, "driverInputs block should exist");

  const block = match[1];
  const weight24 = [...block.matchAll(/weight24:\s*([0-9.]+)/g)].map((m) => Number(m[1]));
  const weight72 = [...block.matchAll(/weight72:\s*([0-9.]+)/g)].map((m) => Number(m[1]));

  assert.equal(weight24.length, weight72.length);
  assert.equal(weight24.length, 12);

  const sum24 = weight24.reduce((sum, value) => sum + value, 0);
  const sum72 = weight72.reduce((sum, value) => sum + value, 0);

  assert.ok(sum24 >= 0.98 && sum24 <= 1.02, `24h weights should remain near normalized (got ${sum24})`);
  assert.ok(sum72 >= 0.98 && sum72 <= 1.02, `72h weights should remain near normalized (got ${sum72})`);
});

test("default FIRMS sources keep VIIRS SNPP first", () => {
  const match = workerSource.match(/const DEFAULT_FIRMS_SOURCES = \[(.*?)\];/s);
  assert.ok(match, "DEFAULT_FIRMS_SOURCES should exist");

  const firstSource = match[1]
    .split(",")
    .map((token) => token.trim().replaceAll('"', ""))
    .filter(Boolean)[0];

  assert.equal(firstSource, "VIIRS_SNPP_NRT");
});

test("frontend fallback and map icon state keep loading as syncing", () => {
  assert.match(appSource, /function fallbackDecisionState\([^)]*\)[\s\S]*return \"SYNCING\";/);
  assert.match(appSource, /SYNCING:\s*\"Syncing Data\"/);
  assert.match(appSource, /const allowed = new Set\(\["safe", "watch-closely", "prepare", "danger-now", "syncing"\]\)/);
  assert.match(appSource, /const normalized = allowed.has\(decisionClass\) \? decisionClass : "syncing"/);
});

test("developer diagnostics raw payload contract is wired in DOM and renderer", () => {
  assert.match(indexSource, /id="diagnostics-raw-meta"/);
  assert.match(indexSource, /id="diagnostics-raw"/);
  assert.match(appSource, /function renderDeveloperDiagnosticsPayload\(/);
  assert.match(appSource, /diagnostics_error:\s*state\.diagnosticsError/);
  assert.match(appSource, /elements\.diagnosticsRaw\.textContent = JSON\.stringify\(payload, null, 2\);/);
});