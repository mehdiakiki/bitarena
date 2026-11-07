pub const SPARSITIES: [f64; 5] = [0.0, 0.5, 0.9, 0.99, 0.999];

pub fn sparsity_label(sparsity: f64) -> String {
    format!("{}%_empty", (sparsity * 100.0) as u32)
}
