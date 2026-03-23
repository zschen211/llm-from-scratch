# CLI 使用说明

本项目在 `cli/` 目录下提供可直接运行的命令行脚本，并在 `cli-tests/` 中配套了基本测试，用于人工快速验证功能。

## BPE Tokenizer

- 训练并导出 checkpoint：`cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py`
- 加载并编码/解码：`cli/llm_from_scratch/bpe_tokenizer/bpe_tokenizer_cli.py`

更多示例见：[`docs/cli/bpe_tokenizer.md`](./bpe_tokenizer.md)

