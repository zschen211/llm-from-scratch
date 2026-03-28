use regex::Regex;
use std::sync::OnceLock;

/// GPT-2 的预分词正则表达式
static ENCODE_SPLIT_PATTERN: OnceLock<Regex> = OnceLock::new();

fn get_pattern() -> &'static Regex {
    ENCODE_SPLIT_PATTERN.get_or_init(|| {
        Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        ).unwrap()
    })
}

/// 预处理和预分词
pub fn preprocess_and_pretokenize(
    text: &str,
    special_tokens: &[String],
) -> Vec<Vec<Vec<u8>>> {
    let mut words = Vec::new();

    // 按 special tokens 切分
    let segments = split_by_special_tokens(text, special_tokens);

    for (kind, segment) in segments {
        if kind == "special" {
            // special token 作为单个 token
            words.push(vec![segment.as_bytes().to_vec()]);
        } else {
            // 普通文本：使用 PAT 正则分词
            let pattern = get_pattern();
            for mat in pattern.find_iter(&segment) {
                let frag = mat.as_str();
                if frag.is_empty() {
                    continue;
                }

                // 检查是否是 special token
                if special_tokens.contains(&frag.to_string()) {
                    words.push(vec![frag.as_bytes().to_vec()]);
                } else {
                    // 拆分为字节
                    let bytes: Vec<Vec<u8>> = frag
                        .as_bytes()
                        .iter()
                        .map(|&b| vec![b])
                        .collect();
                    words.push(bytes);
                }
            }
        }
    }

    words
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
