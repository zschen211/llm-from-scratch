use fancy_regex::Regex;
use std::sync::OnceLock;

/// GPT-2 的预分词正则表达式（tiktoken 对齐）
static TIKTOKEN_GPT2_PATTERN: OnceLock<Regex> = OnceLock::new();

/// CS336 课程的预分词正则表达式
static CS336_PATTERN: OnceLock<Regex> = OnceLock::new();

fn get_tiktoken_pattern() -> &'static Regex {
    TIKTOKEN_GPT2_PATTERN.get_or_init(|| {
        Regex::new(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
        ).expect("Failed to compile tiktoken GPT-2 pattern")
    })
}

fn get_cs336_pattern() -> &'static Regex {
    CS336_PATTERN.get_or_init(|| {
        Regex::new(
            r"'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        ).expect("Failed to compile CS336 pattern")
    })
}

/// 预处理和预分词（使用 fancy-regex 支持环视断言）
pub fn pretokenize_with_pat(
    text: &str,
    special_tokens: &[String],
    use_tiktoken_pat: bool,
) -> Vec<Vec<Vec<u8>>> {
    let mut words = Vec::new();
    let special_set: std::collections::HashSet<String> =
        special_tokens.iter().cloned().collect();

    // 按 special tokens 切分
    let segments = split_by_special_tokens(text, special_tokens);

    // 选择 PAT 模式
    let pattern = if use_tiktoken_pat {
        get_tiktoken_pattern()
    } else {
        get_cs336_pattern()
    };

    for (kind, segment) in segments {
        if kind == "special" {
            // special token 作为单个 token
            words.push(vec![segment.as_bytes().to_vec()]);
        } else {
            // 普通文本：使用 PAT 正则分词
            for mat in pattern.find_iter(&segment) {
                match mat {
                    Ok(m) => {
                        let frag = m.as_str();
                        if frag.is_empty() {
                            continue;
                        }

                        // 检查是否是 special token
                        if special_set.contains(frag) {
                            words.push(vec![frag.as_bytes().to_vec()]);
                        } else {
                            // 将 fragment 拆分为字节级别的 tokens
                            let byte_tokens: Vec<Vec<u8>> = frag
                                .as_bytes()
                                .iter()
                                .map(|&b| vec![b])
                                .collect();
                            words.push(byte_tokens);
                        }
                    }
                    Err(e) => {
                        eprintln!("Regex error: {}", e);
                        break;
                    }
                }
            }
        }
    }

    words
}

/// 预处理和预分词（保留旧接口，使用 tiktoken PAT）
pub fn preprocess_and_pretokenize(
    text: &str,
    special_tokens: &[String],
) -> Vec<Vec<Vec<u8>>> {
    pretokenize_with_pat(text, special_tokens, true)
}

/// 按 special tokens 切分文本
fn split_by_special_tokens(
    text: &str,
    special_tokens: &[String],
) -> Vec<(&'static str, String)> {
    let mut segments = Vec::new();
    let mut specials: Vec<&str> = special_tokens.iter().map(|s| s.as_str()).collect();
    specials.sort_by_key(|s| std::cmp::Reverse(s.len()));

    let special_set: std::collections::HashSet<&str> = specials.iter().copied().collect();

    let mut i = 0;
    let n = text.len();

    while i < n {
        // 尝试匹配 special token
        let mut matched: Option<&str> = None;
        for &st in &specials {
            if text[i..].starts_with(st) {
                matched = Some(st);
                break;
            }
        }

        if let Some(st) = matched {
            segments.push(("special", st.to_string()));
            i += st.len();
            continue;
        }

        // 找到下一个 special token 的位置
        let mut next_sp = n;
        for &st in &specials {
            if let Some(pos) = text[i..].find(st) {
                let abs_pos = i + pos;
                if abs_pos < next_sp {
                    next_sp = abs_pos;
                }
            }
        }

        segments.push(("plain", text[i..next_sp].to_string()));
        i = next_sp;
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_and_pretokenize_basic() {
        let text = "hello world";
        let special_tokens = vec![];

        let words = preprocess_and_pretokenize(text, &special_tokens);

        assert!(!words.is_empty());
        // "hello" 应该被拆分为 5 个字节
        assert_eq!(words[0].len(), 5);
    }

    #[test]
    fn test_preprocess_with_special_tokens() {
        let text = "hello<|endoftext|>world";
        let special_tokens = vec!["<|endoftext|>".to_string()];

        let words = preprocess_and_pretokenize(text, &special_tokens);

        // 应该有 3 个 words: "hello", "<|endoftext|>", "world"
        assert_eq!(words.len(), 3);
        assert_eq!(words[1], vec![b"<|endoftext|>".to_vec()]);
    }
}
