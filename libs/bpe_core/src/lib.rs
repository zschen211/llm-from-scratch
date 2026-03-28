use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

mod merge_optimizer;
mod pair_counter;
mod pretokenizer;

use merge_optimizer::WordsChunkWithIndex;
use pair_counter::count_pairs;
use pretokenizer::preprocess_and_pretokenize;

/// BPE 核心模块的 Python 绑定
#[pymodule]
fn bpe_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    words: &Bound<'_, PyList>,
    min_freq: usize,
) -> PyResult<PyObject> {
    // 转换 Python words 为 Rust 格式
    let rust_words: Vec<Vec<Vec<u8>>> = words
        .iter()
        .map(|word| {
            let word_list: Bound<PyList> = word.downcast_into()?;
            word_list
                .iter()
                .map(|token| {
                    let bytes: Bound<PyBytes> = token.downcast_into()?;
                    Ok(bytes.as_bytes().to_vec())
                })
                .collect::<PyResult<Vec<Vec<u8>>>>()
        })
        .collect::<PyResult<Vec<Vec<Vec<u8>>>>>()?;

    // 调用 Rust 实现
    let pair_counts = count_pairs(&rust_words, min_freq);

    // 转换回 Python dict
    let result = PyDict::new_bound(py);
    for ((left, right), count) in pair_counts {
        let key = PyTuple::new_bound(
            py,
            &[
                PyBytes::new_bound(py, &left).into_any(),
                PyBytes::new_bound(py, &right).into_any(),
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
    let result = PyList::empty_bound(py);
    for word in words {
        let py_word = PyList::empty_bound(py);
        for token in word {
            py_word.append(PyBytes::new_bound(py, &token))?;
        }
        result.append(py_word)?;
    }

    Ok(result.into())
}

/// Python 接口：merge pair 并返回 delta
#[pyfunction]
fn merge_pair_all_words_with_deltas_py(
    py: Python,
    words: &Bound<'_, PyList>,
    left: &Bound<'_, PyBytes>,
    right: &Bound<'_, PyBytes>,
    merged: &Bound<'_, PyBytes>,
) -> PyResult<PyObject> {
    // 转换输入
    let mut rust_words: Vec<Vec<Vec<u8>>> = words
        .iter()
        .map(|word| {
            let word_list: Bound<PyList> = word.downcast_into()?;
            word_list
                .iter()
                .map(|token| {
                    let bytes: Bound<PyBytes> = token.downcast_into()?;
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
        let py_word = PyList::empty_bound(py);
        for token in rust_word {
            py_word.append(PyBytes::new_bound(py, token))?;
        }
        words.set_item(i, py_word)?;
    }

    // 转换 delta 回 Python dict
    let result = PyDict::new_bound(py);
    for ((left, right), count) in delta {
        let key = PyTuple::new_bound(
            py,
            &[
                PyBytes::new_bound(py, &left).into_any(),
                PyBytes::new_bound(py, &right).into_any(),
            ],
        );
        result.set_item(key, count)?;
    }

    Ok(result.into())
}
