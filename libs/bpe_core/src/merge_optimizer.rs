use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

/// 带倒排索引的 words chunk（Rust 实现）
#[pyclass]
pub struct WordsChunkWithIndex {
    words: Vec<Vec<Vec<u8>>>,
    pair_index: FxHashMap<(Vec<u8>, Vec<u8>), FxHashSet<usize>>,
    index_built: bool,
}

#[pymethods]
impl WordsChunkWithIndex {
    #[new]
    fn new(py: Python, words: &PyList) -> PyResult<Self> {
        // 转换 Python words 为 Rust 格式
        let rust_words: Vec<Vec<Vec<u8>>> = words
            .iter()
            .map(|word| {
                let word_list: &PyList = word.extract()?;
                word_list
                    .iter()
                    .map(|token| {
                        let bytes: &PyBytes = token.extract()?;
                        Ok(bytes.as_bytes().to_vec())
                    })
                    .collect::<PyResult<Vec<Vec<u8>>>>()
            })
            .collect::<PyResult<Vec<Vec<Vec<u8>>>>>()?;

        Ok(Self {
            words: rust_words,
            pair_index: FxHashMap::default(),
            index_built: false,
        })
    }

    /// 构建倒排索引
    fn build_index(&mut self) {
        if self.index_built {
            return;
        }

        self.pair_index.clear();

        for (word_idx, word) in self.words.iter().enumerate() {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i + 1].clone());
                self.pair_index
                    .entry(pair)
                    .or_insert_with(FxHashSet::default)
                    .insert(word_idx);
            }
        }

        self.index_built = true;
    }

    /// 使用倒排索引加速 merge
    fn merge_pair_with_deltas(
        &mut self,
        py: Python,
        left: &PyBytes,
        right: &PyBytes,
        merged: &PyBytes,
    ) -> PyResult<PyObject> {
        if !self.index_built {
            self.build_index();
        }

        let left_bytes = left.as_bytes().to_vec();
        let right_bytes = right.as_bytes().to_vec();
        let merged_bytes = merged.as_bytes().to_vec();

        let target_pair = (left_bytes.clone(), right_bytes.clone());

        // 获取受影响的 word indices
        let affected_indices = match self.pair_index.get(&target_pair) {
            Some(indices) => indices.clone(),
            None => return Ok(PyDict::new(py).into()),
        };

        let mut delta: FxHashMap<(Vec<u8>, Vec<u8>), i32> = FxHashMap::default();
        let mut pairs_to_remove: FxHashMap<(Vec<u8>, Vec<u8>), FxHashSet<usize>> =
            FxHashMap::default();
        let mut pairs_to_add: FxHashMap<(Vec<u8>, Vec<u8>), FxHashSet<usize>> =
            FxHashMap::default();

        for &word_idx in &affected_indices {
            let word = &self.words[word_idx];
            if word.len() < 2 {
                continue;
            }

            // 记录旧的 pairs
            let old_pairs: FxHashSet<(Vec<u8>, Vec<u8>)> = (0..word.len() - 1)
                .map(|i| (word[i].clone(), word[i + 1].clone()))
                .collect();

            // 执行 merge
            let (new_word, did_merge) =
                merge_word(word, &left_bytes, &right_bytes, &merged_bytes);

            if !did_merge {
                continue;
            }

            // 计算 delta
            for i in 0..word.len() - 1 {
                if i + 1 < word.len() && word[i] == left_bytes && word[i + 1] == right_bytes {
                    // 记录被移除的 pairs
                    if i > 0 {
                        let pair = (word[i - 1].clone(), word[i].clone());
                        *delta.entry(pair).or_insert(0) -= 1;
                    }
                    *delta.entry(target_pair.clone()).or_insert(0) -= 1;
                    if i + 2 < word.len() {
                        let pair = (word[i + 1].clone(), word[i + 2].clone());
                        *delta.entry(pair).or_insert(0) -= 1;
                    }
                }
            }

            // 计算新增的 pairs
            for i in 0..new_word.len() - 1 {
                if new_word[i] == merged_bytes {
                    if i > 0 {
                        let pair = (new_word[i - 1].clone(), new_word[i].clone());
                        *delta.entry(pair).or_insert(0) += 1;
                    }
                    if i + 1 < new_word.len() {
                        let pair = (new_word[i].clone(), new_word[i + 1].clone());
                        *delta.entry(pair).or_insert(0) += 1;
                    }
                }
            }

            // 更新 words
            self.words[word_idx] = new_word.clone();

            // 记录新的 pairs
            let new_pairs: FxHashSet<(Vec<u8>, Vec<u8>)> = (0..new_word.len() - 1)
                .map(|i| (new_word[i].clone(), new_word[i + 1].clone()))
                .collect();

            // 计算索引更新
            for pair in old_pairs.difference(&new_pairs) {
                pairs_to_remove
                    .entry(pair.clone())
                    .or_insert_with(FxHashSet::default)
                    .insert(word_idx);
            }

            for pair in new_pairs.difference(&old_pairs) {
                pairs_to_add
                    .entry(pair.clone())
                    .or_insert_with(FxHashSet::default)
                    .insert(word_idx);
            }
        }

        // 更新倒排索引
        for (pair, indices) in pairs_to_remove {
            if let Some(index_set) = self.pair_index.get_mut(&pair) {
                for idx in indices {
                    index_set.remove(&idx);
                }
                if index_set.is_empty() {
                    self.pair_index.remove(&pair);
                }
            }
        }

        for (pair, indices) in pairs_to_add {
            self.pair_index
                .entry(pair)
                .or_insert_with(FxHashSet::default)
                .extend(indices);
        }

        // 转换 delta 回 Python dict
        let result = PyDict::new(py);
        for ((left, right), count) in delta {
            if count != 0 {
                let key = pyo3::types::PyTuple::new(
                    py,
                    &[
                        PyBytes::new(py, &left).into(),
                        PyBytes::new(py, &right).into(),
                    ],
                );
                result.set_item(key, count)?;
            }
        }

        Ok(result.into())
    }

    /// 获取 words（用于 Python 访问）
    fn get_words(&self, py: Python) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        for word in &self.words {
            let py_word = PyList::empty(py);
            for token in word {
                py_word.append(PyBytes::new(py, token))?;
            }
            result.append(py_word)?;
        }
        Ok(result.into())
    }
}

/// 在单个 word 中执行 merge
fn merge_word(
    word: &[Vec<u8>],
    left: &[u8],
    right: &[u8],
    merged: &[u8],
) -> (Vec<Vec<u8>>, bool) {
    let mut out = Vec::with_capacity(word.len());
    let mut k = 0;
    let mut did_merge = false;

    while k < word.len() {
        if k + 1 < word.len() && word[k] == left && word[k + 1] == right {
            did_merge = true;
            out.push(merged.to_vec());
            k += 2;
        } else {
            out.push(word[k].clone());
            k += 1;
        }
    }

    (out, did_merge)
}

/// merge_pair_all_words_with_deltas 的纯 Rust 实现（不使用倒排索引）
pub fn merge_pair_all_words_with_deltas(
    words: &mut [Vec<Vec<u8>>],
    left: &[u8],
    right: &[u8],
    merged: &[u8],
) -> HashMap<(Vec<u8>, Vec<u8>), i32> {
    let mut delta: FxHashMap<(Vec<u8>, Vec<u8>), i32> = FxHashMap::default();

    for word in words.iter_mut() {
        if word.len() < 2 {
            continue;
        }

        let (new_word, did_merge) = merge_word(word, left, right, merged);

        if !did_merge {
            continue;
        }

        // 计算 delta（简化版本）
        for i in 0..word.len() - 1 {
            if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
                if i > 0 {
                    let pair = (word[i - 1].clone(), word[i].clone());
                    *delta.entry(pair).or_insert(0) -= 1;
                }
                *delta.entry((left.to_vec(), right.to_vec())).or_insert(0) -= 1;
                if i + 2 < word.len() {
                    let pair = (word[i + 1].clone(), word[i + 2].clone());
                    *delta.entry(pair).or_insert(0) -= 1;
                }
            }
        }

        for i in 0..new_word.len() - 1 {
            if new_word[i] == merged {
                if i > 0 {
                    let pair = (new_word[i - 1].clone(), new_word[i].clone());
                    *delta.entry(pair).or_insert(0) += 1;
                }
                if i + 1 < new_word.len() {
                    let pair = (new_word[i].clone(), new_word[i + 1].clone());
                    *delta.entry(pair).or_insert(0) += 1;
                }
            }
        }

        *word = new_word;
    }

    delta.into_iter().filter(|(_, v)| *v != 0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_word() {
        let word = vec![
            b"h".to_vec(),
            b"e".to_vec(),
            b"l".to_vec(),
            b"l".to_vec(),
            b"o".to_vec(),
        ];

        let (merged_word, did_merge) = merge_word(&word, b"l", b"l", b"ll");

        assert!(did_merge);
        assert_eq!(
            merged_word,
            vec![
                b"h".to_vec(),
                b"e".to_vec(),
                b"ll".to_vec(),
                b"o".to_vec()
            ]
        );
    }

    #[test]
    fn test_merge_word_no_match() {
        let word = vec![b"h".to_vec(), b"e".to_vec(), b"l".to_vec()];

        let (merged_word, did_merge) = merge_word(&word, b"x", b"y", b"xy");

        assert!(!did_merge);
        assert_eq!(merged_word, word);
    }
}
