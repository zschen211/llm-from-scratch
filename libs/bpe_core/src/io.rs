use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Words chunk 的序列化格式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordsChunk {
    pub words: Vec<Vec<Vec<u8>>>,
}

impl WordsChunk {
    pub fn new(words: Vec<Vec<Vec<u8>>>) -> Self {
        Self { words }
    }

    /// 保存到文件（bincode 格式）
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// 从文件加载（bincode 格式）
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let chunk = bincode::deserialize_from(reader)?;
        Ok(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_chunk.bin");

        let words = vec![
            vec![b"h".to_vec(), b"e".to_vec(), b"l".to_vec()],
            vec![b"w".to_vec(), b"o".to_vec(), b"r".to_vec()],
        ];

        let chunk = WordsChunk::new(words.clone());
        chunk.save(&path).unwrap();

        let loaded = WordsChunk::load(&path).unwrap();
        assert_eq!(loaded.words, words);
    }
}
