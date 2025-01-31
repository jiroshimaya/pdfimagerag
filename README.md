# pdfimagerag
PDFを画像に変換してLLMに渡すpythonスクリプトのサンプル。
pymupdfとlangchainを使用。

# Usage
## 環境構築

```
git clone [url]
cd pdfimagerag
```

uvが未インストールの場合インストールしてください。

テスト用に例えば以下のPDFをダウンロードしてプロジェクトルートにおいてください。以下ではjs04report.pdfというファイル名で保存されているとします。

## pdfimagetest.py
PDFを画像に変換してLLMに渡すサンプルコードです。

```
uv run pdfimagetest.py [pdfのファイルパス] [--page_num 画像化するページ番号] [--question LLMに対する質問]
# e.g. uv run pdfimagetest.py js04report.pdf --page_num 0 --question 画像は全体的に何色ですか？
```

## pdfimagerag.py
PDFを文字で検索して画像をLLMに渡すサンプルコードです。

```
uv run pdfimagetest.py [pdfのファイルパス] [質問]
# e.g. uv run pdfimagerag.py js04report.pdf おすすめの観光名所は？
```