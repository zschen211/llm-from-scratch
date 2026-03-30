use crate::io::WordsChunk;
use crate::merge_optimizer::{build_pair_index, merge_global_pair_with_index};
use crate::pair_counter::count_pairs;
use crate::pretokenizer::pretokenize_with_pat;
use log::info;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// 惰性大顶堆：存储 (频次, pair)，与 `pair_counts` 对照剔除过期项
#[derive(Clone, Eq, PartialEq)]
struct PairHeapEntry {
    count: usize,
    pair: (Vec<u8>, Vec<u8>),
}

impl Ord for PairHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for PairHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn pair_heap_pop_best(
    heap: &mut BinaryHeap<PairHeapEntry>,
    pair_counts: &HashMap<(Vec<u8>, Vec<u8>), usize>,
) -> Option<(Vec<u8>, Vec<u8>)> {
    while let Some(e) = heap.pop() {
        if pair_counts.get(&e.pair) == Some(&e.count) && e.count > 0 {
            return Some(e.pair);
        }
    }
    for (pair, &count) in pair_counts {
        if count > 0 {
            heap.push(PairHeapEntry {
                count,
                pair: pair.clone(),
            });
        }
    }
    while let Some(e) = heap.pop() {
        if pair_counts.get(&e.pair) == Some(&e.count) && e.count > 0 {
            return Some(e.pair);
        }
    }
    None
}

fn pair_heap_push_delta(
    heap: &mut BinaryHeap<PairHeapEntry>,
    delta: &HashMap<(Vec<u8>, Vec<u8>), i32>,
    pair_counts: &HashMap<(Vec<u8>, Vec<u8>), usize>,
) {
    for pair in delta.keys() {
        if let Some(&c) = pair_counts.get(pair) {
            if c > 0 {
                heap.push(PairHeapEntry {
                    count: c,
                    pair: pair.clone(),
                });
            }
        }
    }
}

fn pair_heap_fill_from_counts(
    heap: &mut BinaryHeap<PairHeapEntry>,
    pair_counts: &HashMap<(Vec<u8>, Vec<u8>), usize>,
) {
    heap.clear();
    for (pair, &count) in pair_counts {
        if count > 0 {
            heap.push(PairHeapEntry {
                count,
                pair: pair.clone(),
            });
        }
    }
}

fn load_all_words_from_chunks(
    chunk_files: &[PathBuf],
) -> Result<Vec<Vec<Vec<u8>>>, Box<dyn std::error::Error>> {
    let mut all = Vec::new();
    for path in chunk_files {
        let chunk = WordsChunk::load(path)?;
        all.extend(chunk.words);
    }
    Ok(all)
}

fn remove_chunk_files(paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    for p in paths {
        if p.exists() {
            std::fs::remove_file(p)?;
        }
    }
    Ok(())
}

/// 与 Python `llm_from_scratch` 子 logger 对齐，以便 CLI 配置的 file handler 能收到 Rust 侧日志。
const LOG_PRETOKENIZE: &str = "llm_from_scratch.bpe_core.pretokenize";
const LOG_MERGE: &str = "llm_from_scratch.bpe_core.merge";
const LOG_TRAINER: &str = "llm_from_scratch.bpe_core.trainer";

fn mib_u64(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn format_eta_secs(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "?".to_string();
    }
    if seconds < 90.0 {
        format!("{:.0}s", seconds)
    } else if seconds < 3600.0 {
        format!("{:.1}min", seconds / 60.0)
    } else {
        format!("{:.1}h", seconds / 3600.0)
    }
}

/// 所有 chunk 文件在磁盘上的总字节数（merge 阶段每步会读+写一遍，用于 MiB/s 吞吐）。
fn sum_chunk_disk_bytes(paths: &[PathBuf]) -> u64 {
    paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum()
}

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
        // 按字节读入；非法 UTF-8 用 U+FFFD 替换（对齐常见「文本语料含少量坏字节」场景，避免 `read_to_string` 整段报错）。
        let mut raw_chunk: Vec<u8> = Vec::new();
        let mut carry = String::new();
        let mut read_cycles: u64 = 0;
        let mut total_bytes_from_file: u64 = 0;
        let pretokenize_t0 = Instant::now();
        let mut last_progress_log = pretokenize_t0;

        info!(
            target: LOG_PRETOKENIZE,
            "streaming pretokenize start path={} stream_chunk_bytes={} spill_words_threshold=100000 file_size_bytes={}",
            input_path.display(),
            self.stream_chunk_chars,
            file_size
        );

        // 找到最长的 special token 用于对齐
        let split_special = self
            .special_tokens
            .iter()
            .max_by_key(|s| s.len())
            .cloned()
            .unwrap_or_else(|| "<|endoftext|>".to_string());

        loop {
            raw_chunk.clear();
            let bytes_read = reader
                .by_ref()
                .take(self.stream_chunk_chars as u64)
                .read_to_end(&mut raw_chunk)?;

            if bytes_read == 0 && carry.is_empty() {
                break;
            }

            read_cycles += 1;
            total_bytes_from_file = total_bytes_from_file.saturating_add(bytes_read as u64);

            let decoded = String::from_utf8_lossy(&raw_chunk);
            let mut text = if carry.is_empty() {
                decoded.into_owned()
            } else {
                let mut t = std::mem::take(&mut carry);
                t.push_str(&decoded);
                t
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

            // 进度：首包、每 10 次读、EOF 尾块、或至少每 2s 一次（含 MiB/MiB、%、吞吐量）
            let log_this_read = read_cycles == 1
                || read_cycles % 10 == 0
                || (bytes_read == 0 && !text.is_empty())
                || last_progress_log.elapsed() >= Duration::from_secs(2);
            if log_this_read {
                last_progress_log = Instant::now();
                let elapsed = pretokenize_t0.elapsed().as_secs_f64().max(1e-9);
                let mib_read = mib_u64(total_bytes_from_file);
                let mib_s = mib_read / elapsed;
                let (pct_part, total_part) = if file_size > 0 {
                    let pct = 100.0 * (total_bytes_from_file as f64) / (file_size as f64);
                    (
                        format!(" ({:.1}%)", pct),
                        format!("{:.2}MiB/{:.2}MiB", mib_read, mib_u64(file_size)),
                    )
                } else {
                    (String::new(), format!("{:.2}MiB (total size unknown)", mib_read))
                };
                info!(
                    target: LOG_PRETOKENIZE,
                    "pretokenize progress: {}{} throughput={:.2}MiB/s read_cycles={} words_in_memory={} +{}B this_read",
                    total_part,
                    pct_part,
                    mib_s,
                    read_cycles,
                    accumulated_words.len(),
                    bytes_read,
                );
            }

            // 检查是否需要落盘（简化版：每 100K words 落盘一次）
            if accumulated_words.len() >= 100_000 {
                chunk_index += 1;
                let chunk_path = self.chunks_dir.join(format!("chunk_{:06}.bin", chunk_index));
                let n_words = accumulated_words.len();
                let chunk = WordsChunk::new(accumulated_words.clone());
                chunk.save(&chunk_path)?;
                info!(
                    target: LOG_PRETOKENIZE,
                    "spilled chunk {:06} -> {} (words={}, total_chunks={})",
                    chunk_index,
                    chunk_path.display(),
                    n_words,
                    chunk_files.len() + 1,
                );
                chunk_files.push(chunk_path);
                accumulated_words.clear();
            }
        }

        // 保存剩余的 words
        if !accumulated_words.is_empty() {
            chunk_index += 1;
            let chunk_path = self.chunks_dir.join(format!("chunk_{:06}.bin", chunk_index));
            let n_words = accumulated_words.len();
            let chunk = WordsChunk::new(accumulated_words);
            chunk.save(&chunk_path)?;
            info!(
                target: LOG_PRETOKENIZE,
                "spilled final chunk {:06} -> {} (words={}, total_chunks={})",
                chunk_index,
                chunk_path.display(),
                n_words,
                chunk_files.len() + 1,
            );
            chunk_files.push(chunk_path);
        }

        let wall = pretokenize_t0.elapsed().as_secs_f64().max(1e-9);
        info!(
            target: LOG_PRETOKENIZE,
            "pretokenize done chunks={} read_cycles={} total_bytes={} wall={:.1}s avg_throughput={:.2}MiB/s",
            chunk_files.len(),
            read_cycles,
            total_bytes_from_file,
            wall,
            mib_u64(total_bytes_from_file) / wall,
        );

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
        info!(
            target: LOG_TRAINER,
            "train start input={} vocab_size={} num_workers={}",
            input_path.display(),
            self.vocab_size,
            self.num_workers
        );
        let mut chunk_paths = self.streaming_pretokenize(input_path)?;
        let total_chunk_bytes = sum_chunk_disk_bytes(&chunk_paths);
        info!(
            target: LOG_TRAINER,
            "pretokenize finished chunk_files={} total_chunk_on_disk={:.2}MiB ({} bytes)",
            chunk_paths.len(),
            mib_u64(total_chunk_bytes),
            total_chunk_bytes
        );

        // 2. 构建初始 vocab
        let mut vocab = self.build_initial_vocab();
        let mut merges = Vec::new();

        let initial_vocab_size = vocab.len();
        let num_merges = self.vocab_size.saturating_sub(initial_vocab_size);

        info!(
            target: LOG_TRAINER,
            "initial_vocab_size={} planned_merges={}",
            initial_vocab_size,
            num_merges
        );

        // 3. 初始统计
        info!(
            target: LOG_MERGE,
            "counting initial pair frequencies ({} chunks, parallel)...",
            chunk_paths.len()
        );
        let count_t0 = Instant::now();
        let mut pair_counts = self.count_pairs_from_chunks(&chunk_paths)?;
        let count_wall = count_t0.elapsed().as_secs_f64().max(1e-9);
        let count_chunk_mib_s = mib_u64(total_chunk_bytes) / count_wall;
        info!(
            target: LOG_MERGE,
            "initial pair table size={} (count_pairs wall {:.1}s chunk_throughput={:.2}MiB/s over {} chunks)",
            pair_counts.len(),
            count_wall,
            count_chunk_mib_s,
            chunk_paths.len()
        );

        // 4. 迭代 merge：第 1 步读写 chunk；若还有后续步，则加载内存、删 chunk、建倒排索引，之后不再扫盘
        let merge_loop_t0 = Instant::now();
        let mut last_merge_log = merge_loop_t0;
        let mut pair_heap: BinaryHeap<PairHeapEntry> = BinaryHeap::new();
        let mut mem_words: Option<Vec<Vec<Vec<u8>>>> = None;
        let mut mem_pair_index: Option<FxHashMap<(Vec<u8>, Vec<u8>), FxHashSet<usize>>> = None;

        for merge_idx in 0..num_merges {
            if pair_counts.is_empty() {
                info!(
                    target: LOG_MERGE,
                    "no pairs left; stopping early at merge step {}",
                    merge_idx
                );
                break;
            }

            let use_index = mem_words.is_some() && mem_pair_index.is_some();

            // 选择最佳 pair：首步或尚未切到索引模式时用 HashMap；索引模式下用大顶堆 + 惰性校验
            let (left, right) = if use_index {
                match pair_heap_pop_best(&mut pair_heap, &pair_counts) {
                    Some(pair) => pair,
                    None => break,
                }
            } else {
                match self.pick_best_pair(&pair_counts) {
                    Some(pair) => pair,
                    None => break,
                }
            };

            let freq = pair_counts[&(left.clone(), right.clone())];
            let merged = [left.clone(), right.clone()].concat();

            // 添加到 vocab 和 merges
            let new_token_id = initial_vocab_size + merge_idx;
            vocab.insert(new_token_id, merged.clone());
            merges.push((left.clone(), right.clone()));

            let step = merge_idx + 1;

            let merge_step_t0 = Instant::now();
            let delta = if use_index {
                merge_global_pair_with_index(
                    mem_words.as_mut().unwrap(),
                    mem_pair_index.as_mut().unwrap(),
                    &left,
                    &right,
                    &merged,
                )
            } else {
                self.merge_pair_in_chunks(&chunk_paths, &left, &right, &merged)?
            };
            let step_merge_wall = merge_step_t0.elapsed().as_secs_f64().max(1e-9);
            let chunk_throughput_mib_s = if use_index {
                0.0
            } else {
                mib_u64(total_chunk_bytes) / step_merge_wall
            };

            // 更新 pair_counts
            pair_counts.remove(&(left, right));
            for (pair, change) in &delta {
                let count = pair_counts.entry(pair.clone()).or_insert(0);
                *count = (*count as i32 + change).max(0) as usize;
                if *count == 0 {
                    pair_counts.remove(pair);
                }
            }

            if merge_idx == 0 && num_merges > 1 {
                let words = load_all_words_from_chunks(&chunk_paths)?;
                remove_chunk_files(&chunk_paths)?;
                chunk_paths.clear();
                let idx = build_pair_index(&words);
                pair_heap_fill_from_counts(&mut pair_heap, &pair_counts);
                mem_words = Some(words);
                mem_pair_index = Some(idx);
                info!(
                    target: LOG_MERGE,
                    "loaded all words into memory, removed chunk files, built inverted index (distinct_pairs={}) — subsequent merges will not read chunk files",
                    pair_counts.len()
                );
            } else if use_index {
                pair_heap_push_delta(&mut pair_heap, &delta, &pair_counts);
            }

            let elapsed = merge_loop_t0.elapsed().as_secs_f64().max(1e-9);
            let mps = step as f64 / elapsed;
            let pct = 100.0 * (step as f64) / (num_merges as f64).max(1.0);
            let remaining = num_merges.saturating_sub(step);
            let eta_s = if mps > 0.0 {
                remaining as f64 / mps
            } else {
                f64::NAN
            };

            let log_step = step == 1
                || step == num_merges
                || step % 100 == 0
                || num_merges <= 20
                || last_merge_log.elapsed() >= Duration::from_secs(2);
            if log_step {
                last_merge_log = Instant::now();
                let step_merge_rate = 1.0 / step_merge_wall;
                let chunks_on_disk = chunk_paths.len();
                info!(
                    target: LOG_MERGE,
                    "merge progress: {}/{} ({:.1}%) {:.2} merges/s(cum) ETA≈{} step_wall={:.3}s step_merge_rate={:.2}/s chunk_throughput={:.2}MiB/s ({} chunks, {:.2}MiB on disk) inverted_index={} freq={} distinct_pairs={} new_token_id={}",
                    step,
                    num_merges,
                    pct,
                    mps,
                    format_eta_secs(eta_s),
                    step_merge_wall,
                    step_merge_rate,
                    chunk_throughput_mib_s,
                    chunks_on_disk,
                    mib_u64(total_chunk_bytes),
                    use_index,
                    freq,
                    pair_counts.len(),
                    new_token_id
                );
            }
        }

        info!(
            target: LOG_TRAINER,
            "train done final_vocab_size={} merges_applied={}",
            vocab.len(),
            merges.len()
        );

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
