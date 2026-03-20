//! Tokenization for Soprano TTS using HuggingFace tokenizers crate.

use std::path::Path;
use tokenizers::Tokenizer;

/// Special token IDs for Soprano.
pub const TOKEN_UNK: u32 = 0;
pub const TOKEN_TEXT: u32 = 1;
pub const TOKEN_START: u32 = 2;
pub const TOKEN_STOP: u32 = 3;

/// Maximum number of tokens the backbone model accepts.
pub const MAX_TOKENS: usize = 512;

/// Soprano tokenizer wrapper.
pub struct SopranoTokenizer {
    tokenizer: Tokenizer,
}

impl SopranoTokenizer {
    /// Load tokenizer from a tokenizer.json file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    /// Tokenize normalized text (already wrapped with [STOP][TEXT]...[START]).
    /// Returns token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Check if token IDs exceed the model's max sequence length.
    pub fn exceeds_limit(&self, ids: &[u32]) -> bool {
        ids.len() > MAX_TOKENS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_tokenizer_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/tokenizer.json")
    }

    #[test]
    fn test_load_tokenizer() {
        let path = test_tokenizer_path();
        if !path.exists() {
            eprintln!("Skipping test: tokenizer.json not found at {:?}", path);
            return;
        }
        let tok = SopranoTokenizer::from_file(&path).unwrap();
        let ids = tok.encode("[STOP][TEXT]hello world.[START]").unwrap();
        assert_eq!(ids, vec![3, 1, 8077, 8070, 8045, 8004, 8053, 8076, 8139, 8015, 2]);
    }

    #[test]
    fn test_special_tokens() {
        let path = test_tokenizer_path();
        if !path.exists() {
            return;
        }
        let tok = SopranoTokenizer::from_file(&path).unwrap();
        let ids = tok.encode("[STOP][TEXT]mister smith.[START]").unwrap();
        // Verify STOP=3 at start, START=2 at end
        assert_eq!(ids[0], TOKEN_STOP);
        assert_eq!(*ids.last().unwrap(), TOKEN_START);
    }

    #[test]
    fn test_exceeds_limit() {
        let ids = vec![0u32; 513];
        assert!(SopranoTokenizer::exceeds_limit(
            &SopranoTokenizer { tokenizer: Tokenizer::from_file(test_tokenizer_path()).unwrap() },
            &ids
        ));
    }
}
