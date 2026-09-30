# Shared fixtures for the ``bin/perf-report`` tests: a tiny targets
# file and baseline / candidate reports, all in $test_tmpdir.

perf_report_fixtures() {
    cat > "$test_tmpdir/targets.yaml" <<'YAML'
items:
  item-a:
    targets:
      - {cut: c1, metric: calls, target: "<= 1", kind: work}
      - {cut: c1, metric: wait-ms, target: "<= 0.5x", kind: timing}
      - {cut: c1, metric: latency-ms, target: "-100", kind: timing}
    guards:
      - {cut: c2, metric: invariant, target: "== 7", kind: work}
  item-b:
    targets:
      - {cut: c3, metric: flag, target: "== 1", kind: work}
YAML
    cat > "$test_tmpdir/baseline.yaml" <<'YAML'
results:
  item-a:
    c1: {calls: 3, wait-ms: 1000, latency-ms: 400}
    c2: {invariant: 7}
  item-b:
    c3: {flag: 0}
YAML
}
