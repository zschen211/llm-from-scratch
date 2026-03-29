use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// 统计 words 中所有相邻字节对的频率（使用 Rayon 并行）
pub fn count_pairs(
    words: &[Vec<Vec<u8>>],
    min_freq: usize,
) -> HashMap<(Vec<u8>, Vec<u8>), usize> {
    // 使用 Rayon 并行处理每个 word，然后合并结果
    let counts: FxHashMap<(Vec<u8>, Vec<u8>), usize> = words
        .par_iter()
        .fold(
            || FxHashMap::default(),
            |mut acc, word| {
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i].clone(), word[i + 1].clone());
                    *acc.entry(pair).or_insert(0) += 1;
                }
                acc
            },
        )
        .reduce(
            || FxHashMap::default(),
            |mut acc, map| {
                for (pair, count) in map {
                    *acc.entry(pair).or_insert(0) += count;
                }
                acc
            },
        );

    // 过滤低频 pair
    if min_freq > 1 {
        counts
            .into_iter()
            .filter(|(_, count)| *count >= min_freq)
            .collect()
    } else {
        counts.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_pairs_basic() {
        let words = vec![
            vec![b"h".to_vec(), b"e".to_vec(), b"l".to_vec(), b"l".to_vec(), b"o".to_vec()],
            vec![b"w".to_vec(), b"o".to_vec(), b"r".to_vec(), b"l".to_vec(), b"d".to_vec()],
        ];

        let counts = count_pairs(&words, 1);

        assert_eq!(counts.get(&(b"h".to_vec(), b"e".to_vec())), Some(&1));
        assert_eq!(counts.get(&(b"l".to_vec(), b"l".to_vec())), Some(&1));
        assert_eq!(counts.get(&(b"o".to_vec(), b"r".to_vec())), Some(&1));
    }

    #[test]
    fn test_count_pairs_min_freq() {
        let words = vec![
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"c".to_vec(), b"d".to_vec()],
        ];

        let counts = count_pairs(&words, 2);

        assert_eq!(counts.get(&(b"a".to_vec(), b"b".to_vec())), Some(&2));
        assert_eq!(counts.get(&(b"c".to_vec(), b"d".to_vec())), None); // 过滤掉
    }
}
