use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

mod merge_optimizer;
mod pair_counter;
mod pretokenizer;

use merge_optimizer::WordsChunkWithIndex;
use pair_counter::count_pairs;
use pretokenizer::preprocess_and_pretokenize;

/// BPE 核心模块的 Python 绑定
#[pymodule]
fn bpe_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WordsChunkWithIndex>()?;
    m.add_function(wrap_pyfunction!(count_pairs_py, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_and_pretokenize_py, m)?)?;
    m.add_function(wrap_pyfunction!(merge_pair_all_words_with_deltas_py, m)?)?;
    Ok(())
}

/// Python 接口：统计 pair 频率
#[pyfunction]
fn count_pairs_py(
    py: Python,
    words: &PyList,
    min_freq: usize,
) -> PyResult<PyObject> {
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

    // 调用 Rust 实现
    let pair_counts = count_pairs(&rust_words, min_freq);

    // 转换回 Python dict
    let result = PyDict::new(py);
    for ((left, right), count) in pair_counts {
        let key = PyTuple::new(
            py,
            &[
                PyBytes::new(py, &left).into(),
                PyBytes::new(py, &right).into(),
            ],
        );
        result.set_item(key, count)?;
    }

    Ok(result.into())
}

/// Python 接口：预处理和预分词
#[pyfunction]
fn preprocess_and_pretokenize_py(
    py: Python,
    text: &str,
    special_tokens: Vec<String>,
) -> PyResult<PyObject> {
    // 调用 Rust 实现
    let words = preprocess_and_pretokenize(text, &special_tokens);

    // 转换回 Python list[list[bytes]]
    let result = PyList::empty(py);
    for word in words {
        let py_word = PyList::empty(py);
        for token in word {
            py_word.append(PyBytes::new(py, &token))?;
        }
        result.append(py_word)?;
    }

    Ok(result.into())
}

/// Python 接口：merge pair 并返回 delta
#[pyfunction]
fn merge_pair_all_words_with_deltas_py(
    py: Python,
    words: &PyList,
    left: &PyBytes,
    right: &PyBytes,
    merged: &PyBytes,
) -> PyResult<PyObject> {
    // 转换输入
    let mut rust_words: Vec<Vec<Vec<u8>>> = words
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

    let left_bytes = left.as_bytes().to_vec();
    let right_bytes = right.as_bytes().to_vec();
    let merged_bytes = merged.as_bytes().to_vec();

    // 调用 Rust 实现
    let delta = merge_optimizer::merge_pair_all_words_with_deltas(
        &mut rust_words,
        &left_bytes,
        &right_bytes,
        &merged_bytes,
    );

    // 更新 Python words（in-place）
    for (i, rust_word) in rust_words.iter().enumerate() {
        let py_word = PyList::empty(py);
        for token in rust_word {
            py_word.append(PyBytes::new(py, token))?;
        }
        words.set_item(i, py_word)?;
    }

    // 转换 delta 回 Python dict
    let result = PyDict::new(py);
    for ((left, right), count) in delta {
        let key = PyTuple::new(
            py,
            &[
                PyBytes::new(py, &left).into(),
                PyBytes::new(py, &right).into(),
            ],
        );
        result.set_item(key, count)?;
    }

    Ok(result.into())
}
