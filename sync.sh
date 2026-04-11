#!/bin/bash

# 进入笔记目录
cd ~/obsidian保存内容/Notes

# 1. 尝试拉取远端更新，防止多设备冲突
# 使用 --rebase 可以保持提交记录整洁
git pull origin main --rebase

# 2. 检查是否有文件变动
if [ -n "$(git status --porcelain)" ]; then
    echo "检测到变动，准备同步..."
    
    # 添加所有变动
    git add .
    
    # 提交变动，带上时间戳
    git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 3. 推送到远端
    git push origin main
    
    echo "同步完成！"
else
    echo "文件无变动，跳过同步。"
fi
