# PettingLLMs 文档

这个文档使用 MkDocs + Material 主题构建。

## 🚀 快速开始

### 安装依赖

```bash
pip install -r docs/requirements.txt
```

### 本地预览（推荐）

启动本地开发服务器，支持热重载：

```bash
# 方式 1: 使用构建脚本
./build_docs.sh serve

# 方式 2: 直接使用 mkdocs
mkdocs serve
```

然后在浏览器中打开 `http://localhost:8000` 即可查看文档。

### 构建静态网站

生成静态 HTML 文件：

```bash
# 方式 1: 使用构建脚本
./build_docs.sh build

# 方式 2: 直接使用 mkdocs
mkdocs build
```

构建后的文件在 `site/` 目录下。

### 部署到 GitHub Pages

自动部署到 GitHub Pages：

```bash
# 方式 1: 使用构建脚本
./build_docs.sh deploy

# 方式 2: 直接使用 mkdocs
mkdocs gh-deploy
```

### 清理构建文件

```bash
./build_docs.sh clean
```

## 📁 文档结构

```
docs/
├── index.md                    # 首页
├── getting-started/           # 入门指南
│   ├── installation.md
│   ├── quick-start.md
│   └── datasets.md
├── core-concepts/             # 核心概念
│   ├── overview.md
│   ├── at-grpo.md
│   ├── workflows.md
│   └── training-system.md
├── training/                  # 训练指南
│   ├── overview.md
│   ├── games.md
│   ├── planning.md
│   ├── code.md
│   └── math.md
├── evaluation/                # 评估指南
│   └── guide.md
├── results/                   # 结果展示
│   ├── benchmarks.md
│   └── ablations.md
├── api/                       # API 文档
│   └── index.md
└── contributing.md            # 贡献指南
```

## 🎨 特性

- ✨ Material Design 主题
- 🌓 深色/浅色模式切换
- 🔍 全文搜索
- 📱 响应式设计
- 🎯 导航标签页
- 💻 代码高亮
- 📊 MathJax 数学公式支持
- 🔗 自动生成 API 文档

## 📝 编辑文档

1. 所有文档文件使用 Markdown 格式
2. 文档源文件在 `docs/` 目录下
3. 编辑后运行 `mkdocs serve` 可实时预览
4. 主要配置在 `mkdocs.yml` 文件中

## 🔧 配置文件

- `mkdocs.yml` - 主配置文件
- `docs/requirements.txt` - Python 依赖
- `build_docs.sh` - 构建脚本

## 📚 更多信息

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MkDocstrings](https://mkdocstrings.github.io/)
