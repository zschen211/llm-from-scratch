# Merge acceleration based on inverted indices

## 背景
当前 merge 阶段的迭代统计在每轮迭代时都会遍历 chunk 文件进行统计，导致每次 merge 迭代都将大量系统资源和计算开销花费在遍历训练数据上，导致 merge 阶段耗时非常长。因此目前的 merge 流程需要优化