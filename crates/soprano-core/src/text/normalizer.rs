//! Text normalization for Soprano TTS input.
//! Port of soprano/utils/text_normalizer.py

use regex::Regex;
use std::sync::LazyLock;

// ─── Number-to-words engine ────────────────────────────────────────────────

static ONES: &[&str] = &[
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
];

static TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

static ORDINAL_ONES: &[&str] = &[
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth",
];

static ORDINAL_TENS: &[&str] = &[
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth",
    "eightieth", "ninetieth",
];

/// Convert an integer to English words (no "and").
fn number_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    let (neg, n) = if n < 0 { ("minus ", -n as u64) } else { ("", n as u64) };
    format!("{}{}", neg, unsigned_to_words(n))
}

fn unsigned_to_words(n: u64) -> String {
    if n == 0 {
        return String::new();
    }
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let t = TENS[(n / 10) as usize];
        let o = n % 10;
        if o == 0 {
            return t.to_string();
        }
        return format!("{}-{}", t, ONES[o as usize]);
    }
    if n < 1000 {
        let h = ONES[(n / 100) as usize];
        let rem = n % 100;
        if rem == 0 {
            return format!("{} hundred", h);
        }
        return format!("{} hundred {}", h, unsigned_to_words(rem));
    }
    // For thousands and above, Python inflect uses ", " between groups
    if n < 1_000_000 {
        let thousands = n / 1000;
        let rem = n % 1000;
        let t = unsigned_to_words(thousands);
        if rem == 0 {
            return format!("{} thousand", t);
        }
        return format!("{} thousand, {}", t, unsigned_to_words(rem));
    }
    if n < 1_000_000_000 {
        let millions = n / 1_000_000;
        let rem = n % 1_000_000;
        let m = unsigned_to_words(millions);
        if rem == 0 {
            return format!("{} million", m);
        }
        return format!("{} million, {}", m, unsigned_to_words(rem));
    }
    if n < 1_000_000_000_000 {
        let billions = n / 1_000_000_000;
        let rem = n % 1_000_000_000;
        let b = unsigned_to_words(billions);
        if rem == 0 {
            return format!("{} billion", b);
        }
        return format!("{} billion, {}", b, unsigned_to_words(rem));
    }
    let trillions = n / 1_000_000_000_000;
    let rem = n % 1_000_000_000_000;
    let t = unsigned_to_words(trillions);
    if rem == 0 {
        return format!("{} trillion", t);
    }
    format!("{} trillion, {}", t, unsigned_to_words(rem))
}

/// Convert a number to its ordinal word form.
fn number_to_words_ordinal(n: i64) -> String {
    if n == 0 {
        return "zeroth".to_string();
    }
    let abs_n = n.unsigned_abs();
    if abs_n < 20 {
        return ORDINAL_ONES[abs_n as usize].to_string();
    }
    if abs_n < 100 {
        let rem = abs_n % 10;
        if rem == 0 {
            return ORDINAL_TENS[(abs_n / 10) as usize].to_string();
        }
        return format!("{}-{}", TENS[(abs_n / 10) as usize], ORDINAL_ONES[rem as usize]);
    }
    // For larger numbers, convert the cardinal form and add ordinal suffix
    let words = number_to_words(n);
    make_ordinal(&words)
}

/// Convert a cardinal number word string to ordinal.
fn make_ordinal(s: &str) -> String {
    // Replace the last word with its ordinal form
    if s.ends_with("one") {
        format!("{}first", &s[..s.len() - 3])
    } else if s.ends_with("two") {
        format!("{}second", &s[..s.len() - 3])
    } else if s.ends_with("three") {
        format!("{}third", &s[..s.len() - 5])
    } else if s.ends_with("five") {
        format!("{}fifth", &s[..s.len() - 4])
    } else if s.ends_with("eight") {
        format!("{}eighth", &s[..s.len() - 5])
    } else if s.ends_with("nine") {
        format!("{}ninth", &s[..s.len() - 4])
    } else if s.ends_with("twelve") {
        format!("{}twelfth", &s[..s.len() - 6])
    } else if s.ends_with('y') {
        format!("{}ieth", &s[..s.len() - 1])
    } else {
        format!("{}th", s)
    }
}

/// Expand number with special handling for years (1000-3000) matching Python inflect behavior.
fn expand_number(n: i64) -> String {
    if n > 1000 && n < 3000 {
        if n == 2000 {
            return "two thousand".to_string();
        } else if n > 2000 && n < 2010 {
            return format!("two thousand {}", number_to_words((n % 100) as i64));
        } else if n % 100 == 0 {
            return format!("{} hundred", number_to_words((n / 100) as i64));
        } else {
            // group=2 style: "twenty twenty-three" for 2023
            let hi = n / 100;
            let lo = n % 100;
            let hi_words = number_to_words(hi as i64);
            let lo_words = if lo < 10 {
                format!("oh {}", number_to_words(lo as i64))
            } else {
                number_to_words(lo as i64)
            };
            return format!("{} {}", hi_words, lo_words);
        }
    }
    number_to_words(n)
}

// ─── Regex patterns ────────────────────────────────────────────────────────

macro_rules! lazy_regex {
    ($pat:expr) => {
        LazyLock::new(|| Regex::new($pat).unwrap())
    };
    ($pat:expr, $flags:expr) => {
        LazyLock::new(|| regex::RegexBuilder::new($pat).case_insensitive(true).build().unwrap())
    };
}

static NUM_PREFIX_RE: LazyLock<Regex> = lazy_regex!(r"#\d");
static NUM_SUFFIX_RE: LazyLock<Regex> = lazy_regex!(r"\b\d+(K|M|B|T)\b", "i");
static NUM_LETTER_SPLIT_RE: LazyLock<Regex> = lazy_regex!(r"(\d[a-zA-Z]|[a-zA-Z]\d)");
static COMMA_NUMBER_RE: LazyLock<Regex> = lazy_regex!(r"(\d[\d,]+\d)");
static DATE_RE: LazyLock<Regex> = lazy_regex!(r"(^|[^/])(\d\d?[/\-]\d\d?[/\-]\d\d(?:\d\d)?)($|[^/])");
static PHONE_NUMBER_RE: LazyLock<Regex> = lazy_regex!(r"(\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4})");
static TIME_RE: LazyLock<Regex> = lazy_regex!(r"(\d\d?:\d\d(?::\d\d)?)");
static POUNDS_RE: LazyLock<Regex> = lazy_regex!(r"£([\d,]*\d+)");
static DOLLARS_RE: LazyLock<Regex> = lazy_regex!(r"\$([\d.,]*\d+)");
static DECIMAL_NUMBER_RE: LazyLock<Regex> = lazy_regex!(r"(\d+(?:\.\d+)+)");
static MULTIPLY_RE: LazyLock<Regex> = lazy_regex!(r"(\d\s?\*\s?\d)");
static DIVIDE_RE: LazyLock<Regex> = lazy_regex!(r"(\d\s?/\s?\d)");
static ADD_RE: LazyLock<Regex> = lazy_regex!(r"(\d\s?\+\s?\d)");
static SUBTRACT_RE: LazyLock<Regex> = lazy_regex!(r"(\d?\s?-\s?\d)");
static FRACTION_RE: LazyLock<Regex> = lazy_regex!(r"(\d+(?:/\d+)+)");
static ORDINAL_RE: LazyLock<Regex> = lazy_regex!(r"\d+(st|nd|rd|th)");
static NUMBER_RE: LazyLock<Regex> = lazy_regex!(r"\d+");

static LINK_HEADER_RE: LazyLock<Regex> = lazy_regex!(r"(https?://)");
static DASH_RE: LazyLock<Regex> = lazy_regex!(r"(. - .)");
static DOT_RE: LazyLock<Regex> = lazy_regex!(r"([A-Za-z]\.[A-Za-z])");
static PARENTHESES_RE: LazyLock<Regex> = lazy_regex!(r"[\(\[\{].*[\)\]\}](.|$)");
static CAMELCASE_RE: LazyLock<Regex> = lazy_regex!(r"\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b");

static WHITESPACE_RE: LazyLock<Regex> = lazy_regex!(r"\s+");
static SPACE_BEFORE_PUNCT_RE: LazyLock<Regex> = lazy_regex!(r" ([.\?!,])");
static ELLIPSIS_RE: LazyLock<Regex> = lazy_regex!(r"\.\.\.+");
static MULTI_COMMA_RE: LazyLock<Regex> = lazy_regex!(r",+");
static DOT_PUNCT_RE: LazyLock<Regex> = lazy_regex!(r"[\.,]*\.[\.,]*");
static BANG_PUNCT_RE: LazyLock<Regex> = lazy_regex!(r"[\.,!]*![\.,!]*");
static QUESTION_PUNCT_RE: LazyLock<Regex> = lazy_regex!(r"[\.,!\?]*\?[\.,!\?]*");
static ELLIPSIS_PLACEHOLDER_RE: LazyLock<Regex> = lazy_regex!(r"\[ELLIPSIS\]");
// TRIPLE_LETTER_RE uses backreference — handled manually in collapse_triple_letters()
static UNKNOWN_CHARS_RE: LazyLock<Regex> = lazy_regex!(r"[^A-Za-z !\$%&'\*\+,\-./0123456789<>\?_]");
static REMOVE_EXTRA_RE: LazyLock<Regex> = lazy_regex!(r"[<>/_+]");

// ─── Abbreviations ─────────────────────────────────────────────────────────

struct Abbreviation {
    pattern: Regex,
    replacement: &'static str,
}

static ABBREVIATIONS: LazyLock<Vec<Abbreviation>> = LazyLock::new(|| {
    let with_dot: Vec<(&str, &str)> = vec![
        ("mrs", "misess"), ("ms", "miss"), ("mr", "mister"), ("dr", "doctor"),
        ("st", "saint"), ("co", "company"), ("jr", "junior"), ("maj", "major"),
        ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"),
        ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"),
        ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"),
        ("col", "colonel"), ("ft", "fort"),
    ];
    let cased: Vec<(&str, &str)> = vec![
        ("Hz", "hertz"), ("kHz", "kilohertz"),
        ("KBs", "kilobytes"), ("KB", "kilobyte"),
        ("MBs", "megabytes"), ("MB", "megabyte"),
        ("GBs", "gigabytes"), ("GB", "gigabyte"),
        ("TBs", "terabytes"), ("TB", "terabyte"),
        ("APIs", "a p i's"), ("API", "a p i"),
        ("CLIs", "c l i's"), ("CLI", "c l i"),
        ("CPUs", "c p u's"), ("CPU", "c p u"),
        ("GPUs", "g p u's"), ("GPU", "g p u"),
        ("Ave", "avenue"), ("etc", "et cetera"),
        ("Mon", "monday"), ("Tues", "tuesday"), ("Wed", "wednesday"),
        ("Thurs", "thursday"), ("Fri", "friday"), ("Sat", "saturday"),
        ("Jan", "january"), ("Feb", "february"), ("Mar", "march"),
        ("Apr", "april"), ("Aug", "august"), ("Sept", "september"),
        ("Oct", "october"), ("Nov", "november"), ("Dec", "december"),
        ("and/or", "and or"),
    ];

    let mut abbrs = Vec::new();
    for (pat, rep) in with_dot {
        let re = regex::RegexBuilder::new(&format!(r"\b{}\.", regex::escape(pat)))
            .case_insensitive(true)
            .build()
            .unwrap();
        abbrs.push(Abbreviation { pattern: re, replacement: rep });
    }
    for (pat, rep) in cased {
        // Case-sensitive, word boundary
        let re = Regex::new(&format!(r"\b{}\b", regex::escape(pat))).unwrap();
        abbrs.push(Abbreviation { pattern: re, replacement: rep });
    }
    abbrs
});

// ─── Special character replacements ────────────────────────────────────────

struct CharReplacement {
    pattern: Regex,
    replacement: &'static str,
}

static PREUNICODE_SPECIAL: LazyLock<Vec<CharReplacement>> = LazyLock::new(|| {
    vec![CharReplacement {
        pattern: Regex::new("—").unwrap(),
        replacement: " - ",
    }]
});

static SPECIAL_CHARACTERS: LazyLock<Vec<CharReplacement>> = LazyLock::new(|| {
    vec![
        CharReplacement { pattern: Regex::new("@").unwrap(), replacement: " at " },
        CharReplacement { pattern: Regex::new("&").unwrap(), replacement: " and " },
        CharReplacement { pattern: Regex::new("%").unwrap(), replacement: " percent " },
        CharReplacement { pattern: Regex::new(":").unwrap(), replacement: "." },
        CharReplacement { pattern: Regex::new(";").unwrap(), replacement: "," },
        CharReplacement { pattern: Regex::new(r"\+").unwrap(), replacement: " plus " },
        CharReplacement { pattern: Regex::new(r"\\").unwrap(), replacement: " backslash " },
        CharReplacement { pattern: Regex::new("~").unwrap(), replacement: " about " },
        CharReplacement { pattern: Regex::new(r"(^| )<3").unwrap(), replacement: " heart " },
        CharReplacement { pattern: Regex::new("<=").unwrap(), replacement: " less than or equal to " },
        CharReplacement { pattern: Regex::new(">=").unwrap(), replacement: " greater than or equal to " },
        CharReplacement { pattern: Regex::new("<").unwrap(), replacement: " less than " },
        CharReplacement { pattern: Regex::new(">").unwrap(), replacement: " greater than " },
        CharReplacement { pattern: Regex::new("=").unwrap(), replacement: " equals " },
        CharReplacement { pattern: Regex::new("/").unwrap(), replacement: " slash " },
        CharReplacement { pattern: Regex::new("_").unwrap(), replacement: " " },
        CharReplacement { pattern: Regex::new(r"\*").unwrap(), replacement: " " },
    ]
});

// ─── Transform functions ───────────────────────────────────────────────────

fn expand_preunicode(text: &str) -> String {
    let mut result = text.to_string();
    for cr in PREUNICODE_SPECIAL.iter() {
        result = cr.pattern.replace_all(&result, cr.replacement).to_string();
    }
    result
}

fn convert_to_ascii(text: &str) -> String {
    deunicode::deunicode(text)
}

fn normalize_newlines(text: &str) -> String {
    let lines: Vec<String> = text
        .split('\n')
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return String::new();
            }
            if !trimmed.ends_with('.') && !trimmed.ends_with('!') && !trimmed.ends_with('?') {
                format!("{}.", trimmed)
            } else {
                trimmed.to_string()
            }
        })
        .filter(|s| !s.is_empty())
        .collect();
    lines.join(" ")
}

fn normalize_numbers(text: &str) -> String {
    let mut text = text.to_string();

    // #N → "number N"
    text = NUM_PREFIX_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap().as_str();
        format!("number {}", &m[1..])
    }).to_string();

    // NK/M/B/T → "N thousand/million/billion/trillion"
    text = NUM_SUFFIX_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap().as_str();
        let suffix = m.chars().last().unwrap().to_ascii_uppercase();
        let num_part = &m[..m.len() - 1];
        match suffix {
            'K' => format!("{} thousand", num_part),
            'M' => format!("{} million", num_part),
            'B' => format!("{} billion", num_part),
            'T' => format!("{} trillion", num_part),
            _ => m.to_string(),
        }
    }).to_string();

    // Remove commas from numbers
    text = COMMA_NUMBER_RE.replace_all(&text, |caps: &regex::Captures| {
        caps.get(1).unwrap().as_str().replace(',', "")
    }).to_string();

    // Dates
    text = DATE_RE.replace_all(&text, |caps: &regex::Captures| {
        let prefix = caps.get(1).unwrap().as_str();
        let date = caps.get(2).unwrap().as_str();
        let suffix = caps.get(3).unwrap().as_str();
        let parts: Vec<&str> = date.split(|c| c == '/' || c == '-').collect();
        format!("{}{}{}", prefix, parts.join(" dash "), suffix)
    }).to_string();

    // Phone numbers
    text = PHONE_NUMBER_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(1).unwrap().as_str();
        let digits: String = m.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 10 {
            let area: String = digits[..3].chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
            let mid: String = digits[3..6].chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
            let last: String = digits[6..].chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
            format!("{}, {}, {}", area, mid, last)
        } else {
            m.to_string()
        }
    }).to_string();

    // Times
    text = TIME_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(1).unwrap().as_str();
        let parts: Vec<&str> = m.split(':').collect();
        if parts.len() == 2 {
            let hours = parts[0];
            let minutes = parts[1];
            if minutes == "00" {
                let h: i64 = hours.parse().unwrap_or(0);
                if h == 0 {
                    return "0".to_string();
                } else if h <= 12 {
                    return format!("{} o'clock", hours);
                }
                return format!("{} minutes", hours);
            } else if minutes.starts_with('0') {
                return format!("{} oh {}", hours, &minutes[1..]);
            }
            format!("{} {}", hours, minutes)
        } else if parts.len() == 3 {
            let hours = parts[0];
            let minutes = parts[1];
            let seconds = parts[2];
            let h: i64 = hours.parse().unwrap_or(0);
            if h != 0 {
                let min_part = if minutes == "00" {
                    "oh oh".to_string()
                } else if minutes.starts_with('0') {
                    format!("oh {}", &minutes[1..])
                } else {
                    minutes.to_string()
                };
                let sec_part = if seconds == "00" {
                    String::new()
                } else if seconds.starts_with('0') {
                    format!("oh {}", &seconds[1..])
                } else {
                    seconds.to_string()
                };
                format!("{} {} {}", hours, min_part, sec_part).trim().to_string()
            } else if minutes != "00" {
                let sec_part = if seconds == "00" {
                    "oh oh".to_string()
                } else if seconds.starts_with('0') {
                    format!("oh {}", &seconds[1..])
                } else {
                    seconds.to_string()
                };
                format!("{} {}", minutes, sec_part)
            } else {
                seconds.to_string()
            }
        } else {
            m.to_string()
        }
    }).to_string();

    // Pounds
    text = POUNDS_RE.replace_all(&text, "$1 pounds").to_string();

    // Dollars
    text = DOLLARS_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(1).unwrap().as_str();
        let parts: Vec<&str> = m.split('.').collect();
        if parts.len() > 2 {
            return format!("{} dollars", m);
        }
        let dollars: i64 = parts[0].replace(',', "").parse().unwrap_or(0);
        let cents: i64 = if parts.len() > 1 && !parts[1].is_empty() {
            parts[1].parse().unwrap_or(0)
        } else {
            0
        };
        if dollars > 0 && cents > 0 {
            let d_unit = if dollars == 1 { "dollar" } else { "dollars" };
            let c_unit = if cents == 1 { "cent" } else { "cents" };
            format!("{} {}, {} {}", dollars, d_unit, cents, c_unit)
        } else if dollars > 0 {
            let d_unit = if dollars == 1 { "dollar" } else { "dollars" };
            format!("{} {}", dollars, d_unit)
        } else if cents > 0 {
            let c_unit = if cents == 1 { "cent" } else { "cents" };
            format!("{} {}", cents, c_unit)
        } else {
            "zero dollars".to_string()
        }
    }).to_string();

    // Decimal points
    text = DECIMAL_NUMBER_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(1).unwrap().as_str();
        let parts: Vec<&str> = m.split('.').collect();
        let mut result = parts[0].to_string();
        for part in &parts[1..] {
            result.push_str(" point ");
            result.push_str(&part.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" "));
        }
        result
    }).to_string();

    // Math operations
    text = MULTIPLY_RE.replace_all(&text, |caps: &regex::Captures| {
        caps.get(1).unwrap().as_str().split('*').collect::<Vec<_>>().join(" times ")
    }).to_string();

    text = DIVIDE_RE.replace_all(&text, |caps: &regex::Captures| {
        caps.get(1).unwrap().as_str().split('/').collect::<Vec<_>>().join(" over ")
    }).to_string();

    text = ADD_RE.replace_all(&text, |caps: &regex::Captures| {
        caps.get(1).unwrap().as_str().split('+').collect::<Vec<_>>().join(" plus ")
    }).to_string();

    text = SUBTRACT_RE.replace_all(&text, |caps: &regex::Captures| {
        caps.get(1).unwrap().as_str().split('-').collect::<Vec<_>>().join(" minus ")
    }).to_string();

    // Fractions
    text = FRACTION_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(1).unwrap().as_str();
        let parts: Vec<&str> = m.split('/').collect();
        if parts.len() == 2 {
            parts.join(" over ")
        } else {
            parts.join(" slash ")
        }
    }).to_string();

    // Ordinals
    text = ORDINAL_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap().as_str();
        // Strip suffix (st/nd/rd/th) to get the number
        let num_str: String = m.chars().take_while(|c| c.is_ascii_digit()).collect();
        let n: i64 = num_str.parse().unwrap_or(0);
        number_to_words_ordinal(n)
    }).to_string();

    // Split alphanumeric (run twice like Python)
    for _ in 0..2 {
        text = NUM_LETTER_SPLIT_RE.replace_all(&text, |caps: &regex::Captures| {
            let m = caps.get(1).unwrap().as_str();
            let chars: Vec<char> = m.chars().collect();
            format!("{} {}", chars[0], chars[1])
        }).to_string();
    }

    // General numbers
    text = NUMBER_RE.replace_all(&text, |caps: &regex::Captures| {
        let num_str = caps.get(0).unwrap().as_str();
        let n: i64 = num_str.parse().unwrap_or(0);
        expand_number(n)
    }).to_string();

    text
}

fn normalize_special(text: &str) -> String {
    let mut text = text.to_string();

    // URLs
    text = LINK_HEADER_RE.replace_all(&text, "h t t p s colon slash slash ").to_string();

    // Dashes: ". - ." → "X, Y"
    text = DASH_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap().as_str();
        let chars: Vec<char> = m.chars().collect();
        format!("{}, {}", chars[0], chars[4])
    }).to_string();

    // Initials: "A.B" → "A dot B"
    text = DOT_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap().as_str();
        let chars: Vec<char> = m.chars().collect();
        format!("{} dot {}", chars[0], chars[2])
    }).to_string();

    // Parentheses
    text = PARENTHESES_RE.replace_all(&text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap().as_str();
        let re_open = Regex::new(r"[\(\[\{]").unwrap();
        let re_close_mid = Regex::new(r"[\)\]\}][^$.!?,]").unwrap();
        let re_close = Regex::new(r"[\)\]\}]").unwrap();
        let mut result = re_open.replace_all(m, ", ").to_string();
        result = re_close_mid.replace_all(&result, ", ").to_string();
        result = re_close.replace_all(&result, "").to_string();
        result
    }).to_string();

    text
}

fn expand_abbreviations(text: &str) -> String {
    let mut result = text.to_string();
    for abbr in ABBREVIATIONS.iter() {
        result = abbr.pattern.replace_all(&result, abbr.replacement).to_string();
    }
    result
}

static CAMELCASE_SPLIT_RE: LazyLock<Regex> = lazy_regex!("[A-Z][a-z]*");

fn normalize_mixedcase(text: &str) -> String {
    CAMELCASE_RE.replace_all(text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap().as_str();
        let matches: Vec<&str> = CAMELCASE_SPLIT_RE.find_iter(m).map(|mat| mat.as_str()).collect();

        if matches.len() <= 1 {
            return m.to_string(); // Single capital word
        }
        // Python: len(matches) == len(match) means all single-char matches → all uppercase
        if matches.len() == m.len() {
            return m.to_string(); // All uppercase
        }
        // Python: len(matches) == len(match)-1 and ends with 's' → plural uppercase (e.g. "TPUs")
        if matches.len() == m.len() - 1 && m.ends_with('s') {
            return format!("{}'s", &m[..m.len() - 1]); // Plural uppercase
        }
        matches.join(" ")
    }).to_string()
}

fn expand_special_characters(text: &str) -> String {
    let mut result = text.to_string();
    for cr in SPECIAL_CHARACTERS.iter() {
        result = cr.pattern.replace_all(&result, cr.replacement).to_string();
    }
    result
}

fn remove_unknown_characters(text: &str) -> String {
    let result = UNKNOWN_CHARS_RE.replace_all(text, "").to_string();
    REMOVE_EXTRA_RE.replace_all(&result, "").to_string()
}

fn collapse_whitespace(text: &str) -> String {
    let text = WHITESPACE_RE.replace_all(text, " ").to_string();
    let text = SPACE_BEFORE_PUNCT_RE.replace_all(&text, "$1").to_string();
    text.trim().to_string()
}

fn dedup_punctuation(text: &str) -> String {
    let mut text = ELLIPSIS_RE.replace_all(text, "[ELLIPSIS]").to_string();
    text = MULTI_COMMA_RE.replace_all(&text, ",").to_string();
    text = DOT_PUNCT_RE.replace_all(&text, ".").to_string();
    text = BANG_PUNCT_RE.replace_all(&text, "!").to_string();
    text = QUESTION_PUNCT_RE.replace_all(&text, "?").to_string();
    text = ELLIPSIS_PLACEHOLDER_RE.replace_all(&text, "...").to_string();
    text
}

/// Collapse runs of 3+ identical word characters to just 2.
/// e.g. "aaaa" → "aa", "lll" → "ll"
/// Python uses backreference `(\w)\1{2,}` which Rust regex doesn't support.
fn collapse_triple_letters(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        result.push(c);
        if c.is_alphanumeric() || c == '_' {
            // Check if next char is the same
            let mut count = 1;
            while chars.peek() == Some(&c) {
                count += 1;
                chars.next();
                if count <= 2 {
                    result.push(c);
                }
                // If count > 2, we skip (collapse)
            }
        }
    }
    result
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Apply all text cleaning transformations matching Python's `clean_text()`.
pub fn clean_text(text: &str) -> String {
    let text = expand_preunicode(&text);
    let text = convert_to_ascii(&text);
    let text = normalize_newlines(&text);
    let text = normalize_numbers(&text);
    let text = normalize_special(&text);
    let text = expand_abbreviations(&text);
    let text = normalize_mixedcase(&text);
    let text = expand_special_characters(&text);
    let text = text.to_lowercase();
    let text = remove_unknown_characters(&text);
    let text = collapse_whitespace(&text);
    let text = dedup_punctuation(&text);
    let text = collapse_triple_letters(&text);
    text
}

/// Normalize raw text for Soprano TTS input.
/// Cleans text and wraps with special tokens.
pub fn normalize(text: &str) -> String {
    let cleaned = clean_text(text);
    format!("[STOP][TEXT]{}[START]", cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_to_words() {
        assert_eq!(number_to_words(0), "zero");
        assert_eq!(number_to_words(1), "one");
        assert_eq!(number_to_words(13), "thirteen");
        assert_eq!(number_to_words(42), "forty-two");
        assert_eq!(number_to_words(100), "one hundred");
        assert_eq!(number_to_words(123), "one hundred twenty-three");
        assert_eq!(number_to_words(1000), "one thousand");
        assert_eq!(number_to_words(1234), "one thousand, two hundred thirty-four");
    }

    #[test]
    fn test_expand_number_years() {
        assert_eq!(expand_number(2000), "two thousand");
        assert_eq!(expand_number(2001), "two thousand one");
        assert_eq!(expand_number(2023), "twenty twenty-three");
        assert_eq!(expand_number(1999), "nineteen ninety-nine");
        assert_eq!(expand_number(1900), "nineteen hundred");
        assert_eq!(expand_number(1905), "nineteen oh five");
    }

    #[test]
    fn test_ordinals() {
        assert_eq!(number_to_words_ordinal(1), "first");
        assert_eq!(number_to_words_ordinal(2), "second");
        assert_eq!(number_to_words_ordinal(3), "third");
        assert_eq!(number_to_words_ordinal(4), "fourth");
        assert_eq!(number_to_words_ordinal(12), "twelfth");
        assert_eq!(number_to_words_ordinal(20), "twentieth");
        assert_eq!(number_to_words_ordinal(21), "twenty-first");
    }

    // All expected values verified against Python soprano/utils/text_normalizer.py

    #[test]
    fn test_clean_text_basic() {
        assert_eq!(clean_text("Hello, World!"), "hello, world!");
    }

    #[test]
    fn test_clean_text_numbers() {
        assert_eq!(clean_text("$2.47"), "two dollars, forty-seven cents.");
    }

    #[test]
    fn test_clean_text_abbreviations() {
        assert_eq!(clean_text("Mr. Smith"), "mister smith.");
    }

    #[test]
    fn test_normalize_wraps_tokens() {
        assert_eq!(normalize("hello"), "[STOP][TEXT]hello.[START]");
    }

    #[test]
    fn test_clean_text_special_chars() {
        assert_eq!(clean_text("test@email"), "test at email.");
    }

    #[test]
    fn test_phone_number() {
        assert_eq!(
            clean_text("(111) 111-1111"),
            "one one one, one one one, one one one one."
        );
    }

    #[test]
    fn test_time() {
        assert_eq!(clean_text("12:00"), "twelve o'clock.");
        assert_eq!(clean_text("8:05"), "eight oh five.");
    }

    #[test]
    fn test_ordinals_in_text() {
        assert_eq!(clean_text("1st 2nd 3rd 4th"), "first second third fourth.");
    }

    #[test]
    fn test_suffixes() {
        assert_eq!(clean_text("100k"), "one hundred thousand.");
        assert_eq!(clean_text("#1"), "number one.");
    }

    #[test]
    fn test_and_or() {
        assert_eq!(clean_text("and/or"), "and or.");
    }

    #[test]
    fn test_camelcase() {
        assert_eq!(clean_text("LMDeploy"), "l m deploy.");
        assert_eq!(clean_text("LMDeployDecoderModel"), "l m deploy decoder model.");
        assert_eq!(clean_text("Test"), "test.");
        assert_eq!(clean_text("UPPERCASE"), "uppercase.");
        assert_eq!(clean_text("TPUs"), "tpu's.");
    }

    #[test]
    fn test_comma_numbers() {
        assert_eq!(
            clean_text("123,456,789"),
            "one hundred twenty-three million, four hundred fifty-six thousand, seven hundred eighty-nine."
        );
    }

    #[test]
    fn test_dollars_edge() {
        assert_eq!(clean_text("$1.00"), "one dollar.");
        assert_eq!(clean_text("$0.27"), "twenty-seven cents.");
    }

    #[test]
    fn test_decimals() {
        assert_eq!(clean_text("2.47023"), "two point four seven zero two three.");
        assert_eq!(clean_text("1.17.1.1"), "one point one seven point one point one.");
    }

    #[test]
    fn test_dates() {
        assert_eq!(clean_text("1/1/2025"), "one dash one dash twenty twenty-five.");
        assert_eq!(clean_text("A 1/1/11 A"), "a one dash one dash eleven a.");
    }

    #[test]
    fn test_fractions() {
        assert_eq!(clean_text("1/1"), "one over one.");
    }

    #[test]
    fn test_math() {
        assert_eq!(
            clean_text("-1 + 2 * 3 - 4 / 5"),
            "minus one plus two times three minus four over five."
        );
    }

    #[test]
    fn test_initials() {
        assert_eq!(clean_text("U.S.A."), "u dot s.a.");
    }

    #[test]
    fn test_newlines() {
        assert_eq!(clean_text("Hello\nWorld"), "hello. world.");
    }
}
