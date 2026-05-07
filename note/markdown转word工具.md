# Q: 有没有可以将 markdown 转换成精美 word 的工具？

## pandoc (推荐，最灵活)

```bash
# 基本转换
pandoc input.md -o output.docx

# 带数学公式
pandoc input.md -s --wrap=none -o output.docx

# 使用自定义模板
pandoc input.md --reference-doc=template.docx -o output.docx
```

**优点**：效果最好，可精细控制样式
**缺点**：需要配置模板

## 在线工具

- **dillinger.io** - 支持导出 DOCX
- **stackedit.io** - 导出选项丰富
- **md2pdf** - 可以先生成 PDF 再转 Word

## Typora 导出优化

1. 导出后用 Word 打开
2. 应用"设计"主题
3. 调整样式模板（引用块、代码块等）
4. 另存为 `.dotx` 模板

## Python 方案

```python
# 使用 pandoc + 自定义模板
import subprocess
subprocess.run(["pandoc", "input.md", "-o", "output.docx"])
```

## Word 直接打开 MD

Word 2016+ 直接支持打开 Markdown 文件，兼容性不错但样式基础。

---

**推荐**：
1. **pandoc + 自定义模板** — 效果最好，可精细控制
2. 不想折腾用 **dillinger.io** 在线转换
3. Word 打开后手动微调样式