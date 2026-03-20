/// Convert a single f32 PCM sample [-1.0, 1.0] to i16.
#[inline]
pub fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 32767.0) as i16
}

/// Convert a slice of f32 PCM samples to i16 in-place into the output buffer.
/// Returns the number of samples converted.
pub fn f32_slice_to_i16(input: &[f32], output: &mut [i16]) -> usize {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = f32_to_i16(input[i]);
    }
    len
}

/// Convert a slice of f32 PCM samples to a new Vec<i16>.
pub fn f32_to_i16_vec(input: &[f32]) -> Vec<i16> {
    input.iter().map(|&s| f32_to_i16(s)).collect()
}
