from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.prompts import loader


class PromptLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.root.joinpath("sample.md").write_text(
            '开头\n## System\n你好，[[name]]。JSON: {"ok": true}\n## 普通标题\n正文\n## User\n[[question]]',
            encoding="utf-8",
        )
        loader._CACHE.clear()
        self.root_patch = patch.object(loader, "PROMPTS_DIR", self.root)
        self.root_patch.start()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temp_dir.cleanup()

    def test_load_section_and_render_json_braces(self) -> None:
        self.assertEqual(loader.load_prompt("sample.md", "User"), "[[question]]")
        self.assertEqual(
            loader.render_prompt("sample.md", {"name": "小明"}, "System"),
            '你好，小明。JSON: {"ok": true}\n## 普通标题\n正文',
        )

    def test_strict_variables(self) -> None:
        with self.assertRaisesRegex(loader.PromptError, "缺少变量: name"):
            loader.render_prompt("sample.md", {}, "System")
        with self.assertRaisesRegex(loader.PromptError, "未使用变量: extra"):
            loader.render_prompt("sample.md", {"question": "问", "extra": 1}, "User")

    def test_invalid_path_section_and_file(self) -> None:
        for path in ("../outside.md", "sample.txt", "missing.md"):
            with self.subTest(path=path), self.assertRaises(loader.PromptError):
                loader.load_prompt(path)
        with self.assertRaisesRegex(loader.PromptError, "分段不存在"):
            loader.load_prompt("sample.md", "Missing")

    def test_cache_reloads_after_mtime_change(self) -> None:
        self.assertIn("开头", loader.load_prompt("sample.md"))
        time.sleep(0.01)
        self.root.joinpath("sample.md").write_text("已更新", encoding="utf-8")
        self.assertEqual(loader.load_prompt("sample.md"), "已更新")


if __name__ == "__main__":
    unittest.main()
