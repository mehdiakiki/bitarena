#[path = "../benches/support/mod.rs"]
mod benchmark_support;

#[test]
fn sparsity_cases_have_distinct_benchmark_labels() {
    let labels: Vec<String> = benchmark_support::SPARSITIES
        .iter()
        .map(|&sparsity| benchmark_support::sparsity_label(sparsity))
        .collect();
    let unique: std::collections::BTreeSet<&String> = labels.iter().collect();

    assert_eq!(unique.len(), labels.len(), "duplicate labels: {labels:?}");
    assert_eq!(labels.last().unwrap(), "99.9%_empty");
}
