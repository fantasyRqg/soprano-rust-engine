pub mod backbone;
pub mod decoder;
pub mod sampler;
pub mod session;

// Manual Rust-vs-Python reference comparison harness. In-crate (not under
// `tests/`) so it can drive the internal `generate_core` / `run_decoder`
// primitives directly instead of a public one-shot API.
#[cfg(test)]
mod compare_python;
