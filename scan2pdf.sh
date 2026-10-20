#!/bin/bash
# scan2pdf.sh — どこからでも scan2pdf として呼べるようにする起動用スクリプト
src="${BASH_SOURCE[0]}"  # このファイルの場所（リンク経由のことがある）
while [ -L "$src" ]; do  # リンクなら本物の場所までたどる
  dir="$(cd -P "$(dirname "$src")" && pwd)"  # リンクのあるフォルダ
  src="$(readlink "$src")"  # リンクの行き先
  [[ $src != /* ]] && src="$dir/$src"  # 相対パスなら、リンクのあるフォルダを基準にする
done
here="$(cd -P "$(dirname "$src")" && pwd)"  # book_scan のフォルダ
exec "$here/.venv/bin/python" -I "$here/scan2pdf.py" "$@"  # 専用の Python で本体を呼ぶ
