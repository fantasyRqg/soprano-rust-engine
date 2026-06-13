const PREFIX: &str = "[STOP][TEXT]";
const SUFFIX: &str = "[START]";
const TARGET_CHUNK_CHARS: usize = 140;
const MIN_CHUNK_CHARS: usize = 12;
const MAX_MERGED_CHUNK_CHARS: usize = 220;

// Only abbreviations that survive normalization matter here: the normalizer
// expands mr./dr./st./etc. to words and rewrites dotted initials (e.g., u.s.)
// via its letter-dot-letter rule before the chunker ever sees the text.
const ABBREVIATIONS: &[&str] = &["prof.", "sr.", "vs."];

fn wrap_chunk(chunk: &str) -> String {
    format!("{PREFIX}{chunk}{SUFFIX}")
}

fn unwrap_chunk(chunk: &str) -> &str {
    chunk
        .strip_prefix(PREFIX)
        .and_then(|s| s.strip_suffix(SUFFIX))
        .unwrap_or(chunk)
}

fn ends_with_abbreviation(text: &str) -> bool {
    let trimmed = text.trim_end();
    ABBREVIATIONS.iter().any(|abbr| trimmed.ends_with(abbr))
}

/// True when `text` ends with a single-letter initial followed by a period,
/// e.g. "misess j." or "... h." Here the dot abbreviates an initial, not a
/// sentence end, so the chunker must not break a name such as the normalized
/// form of "Mrs. J. H. Riddell" between its initials and surname.
fn ends_with_initial(text: &str) -> bool {
    let Some(before_dot) = text.trim_end().strip_suffix('.') else {
        return false;
    };
    let mut rev = before_dot.chars().rev();
    match rev.next() {
        // The letter must stand alone: preceded by whitespace or nothing,
        // so "h." counts but "riddell." does not.
        Some(c) if c.is_ascii_alphabetic() => match rev.next() {
            None => true,
            Some(prev) => prev.is_whitespace(),
        },
        _ => false,
    }
}

fn should_split_on_soft_boundary(current: &str) -> bool {
    current.trim().len() >= TARGET_CHUNK_CHARS
}

fn should_split_on_strong_boundary(current: &str) -> bool {
    !ends_with_abbreviation(current) && !ends_with_initial(current)
}

fn push_chunk(chunks: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        chunks.push(wrap_chunk(trimmed));
    }
    current.clear();
}

fn merge_short_chunks(chunks: Vec<String>) -> Vec<String> {
    let mut merged: Vec<String> = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let chunk_body = unwrap_chunk(&chunk);
        if let Some(last) = merged.last_mut() {
            let last_body = unwrap_chunk(last);
            let combined_len = last_body.len() + 1 + chunk_body.len();
            if (last_body.len() < MIN_CHUNK_CHARS || chunk_body.len() < MIN_CHUNK_CHARS)
                && combined_len <= MAX_MERGED_CHUNK_CHARS
            {
                let combined = format!("{} {}", last_body.trim_end(), chunk_body.trim_start());
                *last = wrap_chunk(combined.trim());
                continue;
            }
        }
        merged.push(chunk);
    }

    merged
}

/// Split normalized text into sentence-like synthesis chunks.
pub fn chunk_normalized(normalized: &str) -> Vec<String> {
    let body = normalized
        .strip_prefix(PREFIX)
        .and_then(|s| s.strip_suffix(SUFFIX))
        .unwrap_or(normalized)
        .trim();

    if body.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for ch in body.chars() {
        current.push(ch);
        match ch {
            '.' | '!' | '?' if should_split_on_strong_boundary(&current) => {
                push_chunk(&mut chunks, &mut current);
            }
            ';' | ':' | ',' if should_split_on_soft_boundary(&current) => {
                push_chunk(&mut chunks, &mut current);
            }
            _ => {}
        }
    }

    push_chunk(&mut chunks, &mut current);

    if chunks.is_empty() {
        vec![wrap_chunk(body)]
    } else {
        merge_short_chunks(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::chunk_normalized;

    #[test]
    fn splits_multiple_sentences() {
        let normalized = "[STOP][TEXT]hello world. enough of this. last bit![START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "[STOP][TEXT]hello world.[START]");
        assert_eq!(chunks[1], "[STOP][TEXT]enough of this. last bit![START]");
    }

    #[test]
    fn keeps_trailing_text_without_terminal_punctuation() {
        let normalized = "[STOP][TEXT]hello world without stop[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(chunks, vec!["[STOP][TEXT]hello world without stop[START]"]);
    }

    #[test]
    fn keeps_soft_boundaries_inside_short_chunks() {
        let normalized =
            "[STOP][TEXT]enough of this: the story is short; the preamble is shorter.[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(
            chunks,
            vec!["[STOP][TEXT]enough of this: the story is short; the preamble is shorter.[START]"]
        );
    }

    #[test]
    fn splits_long_text_on_soft_boundaries() {
        let normalized = "[STOP][TEXT]this is a fairly long sentence with multiple clauses, enough words to push the chunk toward its target size, and a comma that should become a soft split once the current chunk is long enough, followed by more text to form the second chunk.[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].ends_with(",[START]"));
        assert!(chunks[1].starts_with("[STOP][TEXT]followed by more text"));
    }

    #[test]
    fn does_not_split_common_abbreviations() {
        // prof. survives normalization (unlike dr./mr., which the normalizer
        // expands), so the chunker must not treat its dot as a sentence end.
        let normalized = "[STOP][TEXT]prof. smith arrived. he spoke next.[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "[STOP][TEXT]prof. smith arrived.[START]");
        assert_eq!(chunks[1], "[STOP][TEXT]he spoke next.[START]");
    }

    #[test]
    fn keeps_initials_with_following_surname() {
        // Normalized form of "Mrs. J. H. Riddell wrote ..." — the single-letter
        // initials must not split the name away from the surname.
        let normalized =
            "[STOP][TEXT]misess j. h. riddell wrote many celebrated ghost stories.[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(
            chunks,
            vec!["[STOP][TEXT]misess j. h. riddell wrote many celebrated ghost stories.[START]"]
        );
    }

    #[test]
    fn initials_still_split_at_real_sentence_end() {
        // A surname-terminating period is still a strong boundary.
        let normalized =
            "[STOP][TEXT]misess j. h. riddell wrote ghost stories. he admired her work greatly and often.[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0],
            "[STOP][TEXT]misess j. h. riddell wrote ghost stories.[START]"
        );
        assert_eq!(
            chunks[1],
            "[STOP][TEXT]he admired her work greatly and often.[START]"
        );
    }

    #[test]
    fn keeps_normalized_url_intact() {
        // Normalized form of "... visit https://www.pgdp.net and read ..." — the
        // spelled-out URL contains no terminal punctuation and must stay whole.
        let normalized = "[STOP][TEXT]please visit h t t p s colon slash slash ww dot pgdp dot net and read the guidelines.[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(
            chunks,
            vec!["[STOP][TEXT]please visit h t t p s colon slash slash ww dot pgdp dot net and read the guidelines.[START]"]
        );
    }

    #[test]
    fn merges_short_neighboring_chunks() {
        let normalized = "[STOP][TEXT]hello world. ok? this part is long enough to stand alone as a later chunk because it keeps going for a while and ends clearly.[START]";
        let chunks = chunk_normalized(normalized);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "[STOP][TEXT]hello world. ok?[START]");
        assert_eq!(
            chunks[1],
            "[STOP][TEXT]this part is long enough to stand alone as a later chunk because it keeps going for a while and ends clearly.[START]"
        );
    }
}
