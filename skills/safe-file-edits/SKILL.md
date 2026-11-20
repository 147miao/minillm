---
name: safe-file-edits
description: 修改文件时优先精确编辑并保证原子完整
keywords: [修改文件, edit_file, write_file, 原子替换, 重构]
---

# 安全改文件

1. 先 read_file 看目标文件当前内容；不要凭记忆编辑。
2. 小范围修改优先 edit_file：old_string 必须唯一；若出现多次，扩大上下文或设置 replace_all。
3. 大范围重写用 write_file；宿主会写同目录临时文件再原子替换，读者不会看到半截内容。
4. 改完立刻读回文件，确认关键片段与缩进正确。
5. 能跑测试就跑测试；在最终回答里列出改动文件与验证命令。
6. 不要用 shell 重定向原地覆盖文件；那会绕过原子替换与审计。
