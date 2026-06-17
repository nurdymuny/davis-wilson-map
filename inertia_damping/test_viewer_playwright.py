"""
test_viewer_playwright.py — programmatic verification that the three scenarios
(Quench Up, Baseline, Quench Down) actually drive the cage to visibly different
lift trajectories over a full playback cycle.

Headless Chromium opens cage_preview.html, clicks each scenario button, samples
the live `currentLift` and `displayQ` JavaScript variables every 200ms for ~12
wall-seconds, then prints a per-scenario summary so we can see whether the cage
actually descends during Quench Down or just stays at the top.
"""
from __future__ import annotations

import asyncio
import statistics
import sys
# Force UTF-8 stdout on Windows so the source-tag arrow doesn't crash printing
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
from playwright.async_api import async_playwright


URL = "http://localhost:8000/cage_preview.html"
SAMPLE_INTERVAL_S = 0.2
RECORD_DURATION_S = 12.0
SCENARIOS = ["quench_up", "baseline", "quench_down"]


async def read_dom_state(page) -> dict:
    """Read the values a human sees in the info panel (DOM-only, no sandboxed JS state)."""
    return await page.evaluate(
        """() => {
            const get = (id) => {
                const el = document.getElementById(id);
                return el ? el.textContent.trim() : null;
            };
            const num = (s) => {
                if (s == null) return null;
                const m = String(s).match(/-?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?/);
                return m ? parseFloat(m[0]) : null;
            };
            return {
                q_text: get('q_value'),
                qproxy_text: get('qproxy_value'),
                lift_text: get('lift_value'),
                frame_text: get('frame_value'),
                t_text: get('t_value'),
                source_text: get('source_tag'),
                play_button_text: get('btn_play_pause'),
                Q: num(get('q_value')),
                qProxy: num(get('qproxy_value')),
                lift_m: num(get('lift_value')),
                t_sim: num(get('t_value')),
            };
        }"""
    )


async def measure_scenario(page, scenario: str) -> dict:
    """Click the scenario button, wait, sample DOM readouts."""
    btn_id = f"btn_scen_{scenario}"
    await page.click(f"#{btn_id}")
    # Give the loadScenario async fetch + trajectory parse a moment
    await asyncio.sleep(1.0)
    # Make sure playback is on (Play button is gone; Pause means currently playing)
    pb = await page.evaluate("() => document.getElementById('btn_play_pause').textContent.trim()")
    if pb == 'Play':
        await page.click("#btn_play_pause")
        await asyncio.sleep(0.2)

    samples = []
    n = int(RECORD_DURATION_S / SAMPLE_INTERVAL_S)
    for i in range(n):
        s = await read_dom_state(page)
        s['t_wall'] = i * SAMPLE_INTERVAL_S
        samples.append(s)
        await asyncio.sleep(SAMPLE_INTERVAL_S)

    return {"scenario": scenario, "samples": samples}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        # Listen for console messages
        console_msgs = []
        page.on("console", lambda msg: console_msgs.append(f"[{msg.type}] {msg.text[:200]}"))
        page.on("pageerror", lambda err: console_msgs.append(f"[PAGE ERROR] {err}"))

        print(f"Opening {URL}...", flush=True)
        await page.goto(URL, wait_until="networkidle", timeout=15000)
        await asyncio.sleep(2.0)  # let Three.js scene initialize
        print("Page loaded.", flush=True)

        # Diagnostics on initial state
        initial = await read_dom_state(page)
        print(f"Initial DOM state: source={initial['source_text']}, lift={initial['lift_text']}, Q={initial['q_text']}, play_btn={initial['play_button_text']}", flush=True)

        # Run each scenario
        results = {}
        for scen in SCENARIOS:
            print(f"\n=== Scenario: {scen} (recording {RECORD_DURATION_S}s) ===", flush=True)
            r = await measure_scenario(page, scen)
            results[scen] = r
            # Headline summary right away
            lifts = [s["lift_m"] for s in r["samples"] if s["lift_m"] is not None]
            Qs = [s["Q"] for s in r["samples"] if s["Q"] is not None]
            print(f"  source: {r['samples'][0]['source_text']}", flush=True)
            print(f"  play button shows: {r['samples'][0]['play_button_text']}", flush=True)
            if lifts:
                print(f"  lift_m:  min={min(lifts):.3f}m  max={max(lifts):.3f}m  mean={statistics.mean(lifts):.3f}m  range={max(lifts)-min(lifts):.3f}m", flush=True)
            if Qs:
                print(f"  Q:       min={min(Qs):.3f}     max={max(Qs):.3f}     mean={statistics.mean(Qs):.3f}", flush=True)
            # Print a sparse time-series
            print(f"  t_wall |  Q     |  qProxy  | lift_m  | frame", flush=True)
            for s in r["samples"][::5]:  # every 1s
                q = f"{s['Q']:5.2f}" if s['Q'] is not None else '  —  '
                qp = f"{s['qProxy']:6.4f}" if s['qProxy'] is not None else '  —   '
                lm = f"{s['lift_m']:6.3f}" if s['lift_m'] is not None else '  —   '
                fr = s['frame_text'] or '—'
                print(f"  {s['t_wall']:5.1f}  |  {q} |  {qp}  | {lm} | {fr}", flush=True)
            # Screenshot
            await page.screenshot(path=f"C:/Users/nurdm/OneDrive/Documents/davis-wilson-lattice/inertia_damping/_pw_{scen}.png", full_page=False)

        # Final cross-scenario summary
        print("\n=== CROSS-SCENARIO COMPARISON ===", flush=True)
        print(f"{'scenario':<14} {'lift_min':>10} {'lift_max':>10} {'lift_range':>11} {'Q_min':>8} {'Q_max':>8}", flush=True)
        for scen, r in results.items():
            lifts = [s["lift_m"] for s in r["samples"] if s["lift_m"] is not None]
            Qs = [s["Q"] for s in r["samples"] if s["Q"] is not None]
            if lifts and Qs:
                print(f"{scen:<14} {min(lifts):>10.3f} {max(lifts):>10.3f} {max(lifts)-min(lifts):>11.3f} {min(Qs):>8.3f} {max(Qs):>8.3f}", flush=True)
            else:
                print(f"{scen:<14} NO DATA — viewer may be on synthetic fallback", flush=True)

        # Console diagnostics
        if console_msgs:
            print(f"\n=== Browser console messages ({len(console_msgs)}) ===", flush=True)
            for m in console_msgs[:20]:
                print(f"  {m}", flush=True)

        await browser.close()


# ---------------------------------------------------------------------------
# Modal-schema-compat test (patch 14, Stage 2)
#
# Verify that the cage_preview audit modal renders schema 1.0, 1.1, and 1.2
# report fixtures without JavaScript errors, with the correct category-row
# count, and without spurious "schema mismatch" warnings for the older
# schemas (whose overall_verdict.total still matches their own rendered
# count).
# ---------------------------------------------------------------------------
import os as _os
import json as _json
from pathlib import Path as _Path

FIXTURE_DIR = _Path(__file__).parent / "test_fixtures"
SCHEMA_FIXTURES = [
    {"schema": "1.0", "report": "report_schema_1_0.json", "manifest": "manifest_schema_1_0.json",
     "expects_section_8": False, "expects_section_9": False, "expected_total": 7},
    {"schema": "1.1", "report": "report_schema_1_1.json", "manifest": "manifest_schema_1_1.json",
     "expects_section_8": False, "expects_section_9": False, "expected_total": 8},
    {"schema": "1.2", "report": "report_schema_1_2.json", "manifest": "manifest_schema_1_2.json",
     "expects_section_8": True,  "expects_section_9": True,  "expected_total": 10},
]


async def _verify_fixture(p, browser_ctx, url: str, fixture: dict) -> dict:
    """Open cage_preview.html with the given fixture injected via page.route()."""
    page = await browser_ctx.new_page()
    pageerrors: list[str] = []
    console_errors: list[str] = []
    page.on("pageerror", lambda err: pageerrors.append(str(err)))
    page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)

    # Intercept reports/latest/{manifest,report}.json with the fixture content.
    report_blob = (FIXTURE_DIR / fixture["report"]).read_text(encoding="utf-8")
    manifest_blob = (FIXTURE_DIR / fixture["manifest"]).read_text(encoding="utf-8")

    async def handle_report(route):
        await route.fulfill(status=200, content_type="application/json", body=report_blob)
    async def handle_manifest(route):
        await route.fulfill(status=200, content_type="application/json", body=manifest_blob)

    await page.route("**/reports/latest/report.json", handle_report)
    await page.route("**/reports/latest/manifest.json", handle_manifest)

    await page.goto(url, wait_until="networkidle", timeout=15000)
    # Open the audit modal
    await page.click("#btn_generate_report")
    # Wait for the modal to populate (renderer is async)
    await asyncio.sleep(1.0)
    # Wait for any retries / xhr settle
    await page.wait_for_function(
        "() => document.getElementById('audit_modal') && document.getElementById('audit_modal').classList.contains('open')",
        timeout=8000,
    )
    await asyncio.sleep(0.5)

    # Count category rows in the verdict summary table. The renderer puts these
    # in a <table> with rows of the form <tr><td>Category</td><td>VERDICT</td></tr>
    # under the "Verdict summary" section_head.
    summary = await page.evaluate(
        """() => {
            const body = document.getElementById('audit_modal_body');
            if (!body) return { error: 'no audit_modal_body' };
            // Find Section 8 / 9 markers
            const html = body.innerHTML.toLowerCase();
            const hasSection8 = html.includes('operational beta envelope (section 8') ||
                                html.includes('section 8') && html.includes('beta envelope');
            const hasSection9 = html.includes('sector classifier') &&
                                (html.includes('section 9') || html.includes('band discrimination'));
            const hasSchemaMismatchWarning = html.includes('schema mismatch');
            // Count category rows: rows where the second cell has a class starting with 'verdict_'.
            const rows = Array.from(body.querySelectorAll('table tr'));
            const verdictRows = rows.filter(r => {
                const cells = r.querySelectorAll('td');
                if (cells.length < 2) return false;
                const cls = cells[1].className || '';
                return cls.indexOf('verdict_') === 0;
            });
            return {
                verdict_row_count: verdictRows.length,
                hasSection8, hasSection9, hasSchemaMismatchWarning,
                bodyText: body.innerText.substring(0, 1200),
            };
        }"""
    )

    await page.close()
    return {
        "fixture": fixture,
        "summary": summary,
        "pageerrors": pageerrors,
        "console_errors": console_errors,
    }


async def test_modal_schema_compat(url: str = URL) -> int:
    """Drive the cage_preview audit modal against schema 1.0 / 1.1 / 1.2 fixtures.

    Returns 0 on full pass, non-zero on any check failure.
    """
    print("=" * 68, flush=True)
    print("test_modal_schema_compat (Stage 2 patch 14)", flush=True)
    print("=" * 68, flush=True)
    print(f"URL: {url}", flush=True)
    print(f"Fixtures: {FIXTURE_DIR}", flush=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        try:
            results = []
            for fixture in SCHEMA_FIXTURES:
                print(f"\n--- schema {fixture['schema']} -> expect rows={fixture['expected_total']}, "
                      f"s8={fixture['expects_section_8']}, s9={fixture['expects_section_9']}",
                      flush=True)
                r = await _verify_fixture(pw, ctx, url, fixture)
                results.append(r)

                s = r["summary"]
                ok_count = (s.get("verdict_row_count") == fixture["expected_total"])
                ok_s8 = (s.get("hasSection8") is fixture["expects_section_8"])
                ok_s9 = (s.get("hasSection9") is fixture["expects_section_9"])
                ok_no_mismatch = (s.get("hasSchemaMismatchWarning") is False)
                ok_no_pageerrors = (len(r["pageerrors"]) == 0)
                ok_no_console_errors = (len(r["console_errors"]) == 0)
                print(f"  row_count={s.get('verdict_row_count')} (want {fixture['expected_total']}) -> {'PASS' if ok_count else 'FAIL'}", flush=True)
                print(f"  hasSection8={s.get('hasSection8')} (want {fixture['expects_section_8']}) -> {'PASS' if ok_s8 else 'FAIL'}", flush=True)
                print(f"  hasSection9={s.get('hasSection9')} (want {fixture['expects_section_9']}) -> {'PASS' if ok_s9 else 'FAIL'}", flush=True)
                print(f"  schema-mismatch warning absent: {'PASS' if ok_no_mismatch else 'FAIL'}", flush=True)
                print(f"  no JS pageerror: {'PASS' if ok_no_pageerrors else 'FAIL ' + str(r['pageerrors'])}", flush=True)
                print(f"  no console errors: {'PASS' if ok_no_console_errors else 'FAIL ' + str(r['console_errors'])}", flush=True)

                r["ok"] = all([ok_count, ok_s8, ok_s9, ok_no_mismatch, ok_no_pageerrors, ok_no_console_errors])
        finally:
            await browser.close()

    fails = [r for r in results if not r.get("ok")]
    print("\n" + "=" * 68, flush=True)
    if fails:
        print(f"OVERALL: {len(fails)}/{len(results)} fixtures FAILED", flush=True)
        return 2
    print(f"OVERALL: {len(results)}/{len(results)} schema fixtures PASS", flush=True)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-compat-only", action="store_true",
                    help="run only the modal-schema-compat suite (patch 14, Stage 2)")
    ap.add_argument("--url", default=URL, help="cage_preview.html URL")
    args, _rest = ap.parse_known_args()

    if args.schema_compat_only:
        sys.exit(asyncio.run(test_modal_schema_compat(args.url)))
    asyncio.run(main())
