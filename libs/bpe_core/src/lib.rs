use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

mod merge_optimizer;
mod pair_counter;
mod pretokenizer;
mod io;
mod trainer;

use merge_optimizer::WordsChunkWithIndex;
use pair_counter::count_pairs;
use pretokenizer::{preprocess_and_pretokenize, pretokenize_with_pat};
use io::WordsChunk;
use trainer::BPETrainer;

/// BPE 核心模块的 Python 绑定
#[pymodule]
fn bpe_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WordsChunkWithIndex>()?;
    m.add_function(wrap_pyfunction!(count_pairs_py, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_and_pretokenize_py, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize_with_pat_py, m)?)?;
    m.add_function(wrap_pyfunction!(merge_pair_all_words_with_deltas_py, m)?)?;
    m.add_function(wrap_pyfunction!(dump_words_chunk_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_words_chunk_py, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe_full_py, m)?)?;
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

/// Python 接口：使用 fancy-regex 的预分词（支持环视断言）
#[pyfunction]
fn pretokenize_with_pat_py(
    py: Python,
    text: &str,
    special_tokens: Vec<String>,
    use_tiktoken_pat: bool,
) -> PyResult<PyObject> {
    // 调用 Rust 实现
    let words = pretokenize_with_pat(text, &special_tokens, use_tiktoken_pat);

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

/// Python 接口：保存 words chunk 到文件（bincode 格式）
#[pyfunction]
fn dump_words_chunk_py(
    py: Python,
    path: &str,
    words: &Bound<'_, PyList>,
) -> PyResult<()> {
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

    let chunk = WordsChunk::new(rust_words);
    chunk.save(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to save chunk: {}", e))
    })?;

    Ok(())
}

/// Python 接口：从文件加载 words chunk（bincode 格式）
#[pyfunction]
fn load_words_chunk_py(
    py: Python,
    path: &str,
) -> PyResult<PyObject> {
    let chunk = WordsChunk::load(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to load chunk: {}", e))
    })?;

    // 转换回 Python list[list[bytes]]
    let result = PyList::empty_bound(py);
    for word in chunk.words {
        let py_word = PyList::empty_bound(py);
        for token in word {
            py_word.append(PyBytes::new_bound(py, &token))?;
        }
        result.append(py_word)?;
    }

    Ok(result.into())
}

/// Python 接口：完整的 BPE 训练流程
#[pyfunction]
fn train_bpe_full_py(
    py: Python,
    input_path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
    num_workers: usize,
    stream_chunk_chars: usize,
    chunks_dir: &str,
) -> PyResult<PyObject> {
    use std::path::PathBuf;

    let trainer = BPETrainer::new(
        special_tokens,
        vocab_size,
        num_workers,
        stream_chunk_chars,
        PathBuf::from(chunks_dir),
    );

    let (vocab, merges) = trainer
        .train(&PathBuf::from(input_path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Training failed: {}", e)))?;

    // 转换 vocab 为 Python dict
    let py_vocab = PyDict::new_bound(py);
    for (idx, token) in vocab {
        py_vocab.set_item(idx, PyBytes::new_bound(py, &token))?;
    }

    // 转换 merges 为 Python list
    let py_merges = PyList::empty_bound(py);
    for (left, right) in merges {
        let pair = PyTuple::new_bound(
            py,
            &[
                PyBytes::new_bound(py, &left).into_any(),
                PyBytes::new_bound(py, &right).into_any(),
            ],
        );
        py_merges.append(pair)?;
    }

    // 返回 (vocab, merges) tuple
    let result = PyTuple::new_bound(py, &[py_vocab.into_any(), py_merges.into_any()]);
    Ok(result.into())
}
