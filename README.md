# 全唐诗智能匹配系统

基于 CLIP 的多模态检索原型，支持以图寻诗、文本搜诗和相似诗推荐。后端读取全唐诗 JSON 数据，提取文本/图像向量后通过余弦相似度返回 Top-N 结果。

## Files

- `后端.py`: Flask API 和 CLIP 特征检索逻辑。
- `前端.html`: 单页前端界面。
- `poet.tang.0.json`, `poet.tang.1000.json`: 示例全唐诗数据分片。
- `代码运行说明.txt`: 原始运行环境说明。

## Run

```bash
pip install torch transformers pillow numpy scikit-learn flask flask-cors
python 后端.py
```

然后打开 `前端.html`。
