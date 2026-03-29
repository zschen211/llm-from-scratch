use crate::io::WordsChunk;
use crate::pair_counter::count_pairs;
use crate::pretokenizer::pretokenize_with_pat;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// BPE 训练器
pub struct BPETrainer {
    special_tokens: Vec<String>,
    vocab_size: usize,
    num_workers: usize,
    stream_chunk_chars: usize,
    chunks_dir: PathBuf,
}

impl BPETrainer {
    pub fn new(
        special_tokens: Vec<String>,
        vocab_size: usize,
        num_workers: usize,
        stream_chunk_chars: usize,
        chunks_dir: PathBuf,
    ) -> Self {
        Self {
            special_tokens,
            vocab_size,
            num_workers,
            stream_chunk_chars,
            chunks_dir,
        }
    }

    /// 流式预分词：读取文件并生成 chunk 文件
    pub fn streaming_pretokenize(
        &self,
        input_path: &Path,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&self.chunks_dir)?;

        let file = File::open(input_path)?;
        let file_size = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        let mut chunk_files = Vec::new();
        let mut chunk_index = 0;
        let mut accumulated_words = Vec::new();
        let mut buffer = String::new();
        let mut carry = String::new();

        // 找到最长的 special token 用于对齐
        let split_special = self
            .special_tokens
            .iter()
            .max_by_key(|s| s.len())
            .cloned()
            .unwrap_or_else(|| "<|endoftext|>".to_string());

        loop {
            buffer.clear();
            let bytes_read = reader
                .by_ref()
                .take(self.stream_chunk_chars as u64)
                .read_to_string(&mut buffer)?;

            if bytes_read == 0 && carry.is_empty() {
                break;
            }

            let mut text = if !carry.is_empty() {
                carry.clone() + &buffer
            } else {
                buffer.clone()
            };

            // 在 special token 边界对齐
            if bytes_read > 0 {
                if let Some(pos) = text.rfind(&split_special) {
                    let cut = pos + split_special.len();
                    if cut < text.len() {
                        carry = text[cut..].to_string();
                        text = text[..cut].to_string();
                    } else {
                        carry.clear();
                    }
                } else {
                    carry.clear();
                }
            } else {
                carry.clear();
            }

            if text.is_empty() {
                continue;
            }

            // 预分词
            let words_part = pretokenize_with_pat(&text, &self.special_tokens, true);
            accumulated_words.extend(words_part);

            // 检查是否需要落盘（简化版：每 100K words 落盘一次）
            if accumulated_words.len() >= 100_000 {
                chunk_index += 1;
                let chunk_path = self.chunks_dir.join(format!("chunk_{:06}.bin", chunk_index));
                let chunk = WordsChunk::new(accumulated_words.clone());
                chunk.save(&chunk_path)?;
                chunk_files.push(chunk_path);
                accumulated_words.clear();
            }
        }

        // 保存剩余的 words
        if !accumulated_words.is_empty() {
            chunk_index += 1;
            let chunk_path = self.chunks_dir.join(format!("chunk_{:06}.bin", chunk_index));
            let chunk = WordsChunk::new(accumulated_words);
            chunk.save(&chunk_path)?;
            chunk_files.push(chunk_path);
        }

        Ok(chunk_files)
    }

    /// 从 chunk 文件统计 pair 频率
    fn count_pairs_from_chunks(
        &self,
        chunk_files: &[PathBuf],
    ) -> Result<HashMap<(Vec<u8>, Vec<u8>), usize>, Box<dyn std::error::Error>> {
        let counts: HashMap<(Vec<u8>, Vec<u8>), usize> = chunk_files
            .par_iter()
            .map(|path| {
                let chunk = WordsChunk::load(path).unwrap();
                count_pairs(&chunk.words, 1)
            })
            .reduce(
                || HashMap::default(),
                |mut acc, map| {
                    for (pair, count) in map {
                        *acc.entry(pair).or_insert(0) += count;
                    }
                    acc
                },
            );

        Ok(counts)
    }

    /// 在所有 chunk 文件中执行 merge
    fn merge_pair_in_chunks(
        &self,
        chunk_files: &[PathBuf],
        left: &[u8],
        right: &[u8],
        merged: &[u8],
    ) -> Result<HashMap<(Vec<u8>, Vec<u8>), i32>, Box<dyn std::error::Error>> {
        use crate::merge_optimizer::merge_pair_all_words_with_deltas;

        let total_delta: HashMap<(Vec<u8>, Vec<u8>), i32> = chunk_files
            .par_iter()
            .map(|path| {
                let mut chunk = WordsChunk::load(path).unwrap();
                let delta = merge_pair_all_words_with_deltas(&mut chunk.words, left, right, merged);
                chunk.save(path).unwrap();
                delta
            })
            .reduce(
                || HashMap::default(),
                |mut acc, map| {
                    for (pair, count) in map {
                        *acc.entry(pair).or_insert(0) += count;
                    }
                    acc
                },
            );

        Ok(total_delta
            .into_iter()
            .filter(|(_, v)| *v != 0)
            .collect())
    }

    /// 选择频率最高的 pair
    fn pick_best_pair(
        &self,
        pair_counts: &HashMap<(Vec<u8>, Vec<u8>), usize>,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        if pair_counts.is_empty() {
            return None;
        }

        let max_freq = *pair_counts.values().max()?;
        let candidates: Vec<_> = pair_counts
            .iter()
            .filter(|(_, &freq)| freq == max_freq)
            .map(|(pair, _)| pair.clone())
            .collect();

        candidates.into_iter().max()
    }

    /// 构建初始 vocab
    fn build_initial_vocab(&self) -> HashMap<usize, Vec<u8>> {
        let mut vocab = HashMap::new();
        let mut idx = 0;

        // 添加 special tokens
        for st in &self.special_tokens {
            vocab.insert(idx, st.as_bytes().to_vec());
            idx += 1;
        }

        // 添加 GPT-2 字节编码
        for b in gpt2_byte_positions() {
            vocab.insert(idx, vec![b]);
            idx += 1;
        }

        vocab
    }

    /// 完整的训练流程
    pub fn train(
        &self,
        input_path: &Path,
    ) -> Result<(HashMap<usize, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), Box<dyn std::error::Error>>
    {
        // 1. 流式预分词
        println!("开始流式预分词...");
        let chunk_files = self.streaming_pretokenize(input_path)?;
        println!("预分词完成，生成 {} 个 chunk 文件", chunk_files.len());

        // 2. 构建初始 vocab
        let mut vocab = self.build_initial_vocab();
        let mut merges = Vec::new();

        let initial_vocab_size = vocab.len();
        let num_merges = self.vocab_size.saturating_sub(initial_vocab_size);

        println!("初始 vocab 大小: {}", initial_vocab_size);
        println!("需要执行 {} 次 merge", num_merges);

        // 3. 初始统计
        println!("统计初始 pair 频率...");
        let mut pair_counts = self.count_pairs_from_chunks(&chunk_files)?;
        println!("初始 pair 数量: {}", pair_counts.len());

        // 4. 迭代 merge
        for merge_idx in 0..num_merges {
            if pair_counts.is_empty() {
                println!("没有更多 pair 可以合并，提前结束");
                break;
            }

            // 选择最佳 pair
            let (left, right) = match self.pick_best_pair(&pair_counts) {
                Some(pair) => pair,
                None => break,
            };

            let freq = pair_counts[&(left.clone(), right.clone())];
            let merged = [left.clone(), right.clone()].concat();

            // 添加到 vocab 和 merges
            let new_token_id = initial_vocab_size + merge_idx;
            vocab.insert(new_token_id, merged.clone());
            merges.push((left.clone(), right.clone()));

            if merge_idx % 100 == 0 {
                println!(
                    "Merge {}/{}: freq={}, pair_len={}",
                    merge_idx + 1,
                    num_merges,
                    freq,
                    pair_counts.len()
                );
            }

            // 执行 merge 并获取增量
            let delta = self.merge_pair_in_chunks(&chunk_files, &left, &right, &merged)?;

            // 更新 pair_counts
            pair_counts.remove(&(left, right));
            for (pair, change) in delta {
                let count = pair_counts.entry(pair.clone()).or_insert(0);
                *count = (*count as i32 + change).max(0) as usize;
                if *count == 0 {
                    pair_counts.remove(&pair);
                }
            }
        }

        println!("训练完成！最终 vocab 大小: {}", vocab.len());

        Ok((vocab, merges))
    }
}

/// GPT-2 字节编码位置
fn gpt2_byte_positions() -> Vec<u8> {
    let mut bytes = Vec::new();
    // 可打印 ASCII
    for b in b'!'..=b'~' {
        bytes.push(b);
    }
    bytes.push(b' ');
    // 扩展 ASCII
    for b in 161..=172 {
        bytes.push(b);
    }
    for b in 174..=255 {
        bytes.push(b);
    }
    // 剩余字节映射到高位
    let mut remaining = Vec::new();
    for b in 0..=255u8 {
        if !bytes.contains(&b) {
            remaining.push(b);
        }
    }
    bytes.extend(remaining);
    bytes
}
