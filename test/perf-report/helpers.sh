# shellcheck shell=bash
# Shared fixtures for the ``bin/perf-report`` tests: a tiny targets
# file and baseline / candidate reports, all in $test_tmpdir.
#
# $test_tmpdir and $base are provided by sunit / bin/test-perf-report.
# shellcheck disable=SC2154

perf_report_fixtures() {
    cat > "$test_tmpdir/targets.yaml" <<'YAML'
items:
  item-a:
    cuts: [c1, c2]
    tests: [{scope: lib, filter: "a::"}]
    prerequisites: [{scope: lib, filter: "a_pre::"}]
    targets:
      - {cut: c1, metric: calls, target: "<= 1", kind: work}
      - {cut: c1, metric: wait-ms, target: "<= 0.5x", kind: timing}
      - {cut: c1, metric: latency-ms, target: "-100", kind: timing}
      - {cut: c1, metric: extra, target: "<= 10", kind: work, advisory: true}
    guards:
      - {cut: c2, metric: invariant, target: "== 7", kind: work}
  item-b:
    cuts: [c3]
    tests: [{scope: perf, filter: "b::"}]
    targets:
      - {cut: c3, metric: flag, target: "== 1", kind: work}
YAML
    cat > "$test_tmpdir/baseline.yaml" <<'YAML'
results:
  item-a:
    c1: {calls: 3, wait-ms: 1000, latency-ms: 400, extra: 2}
    c2: {invariant: 7}
  item-b:
    c3: {flag: 0}
YAML
}

# Write a candidate report: $1 = runs YAML list, stdin = results YAML.
perf_report_candidate() {
    {
        printf 'runs: %s\n' "$1"
        cat
    } > "$test_tmpdir/now.yaml"
}

perf_report_compare() {
    NO_COLOR=1 "$base/bin/perf-report" --targets "$test_tmpdir/targets.yaml" \
        --compare "$test_tmpdir/baseline.yaml" --report "$test_tmpdir/now.yaml" "$@"
}
