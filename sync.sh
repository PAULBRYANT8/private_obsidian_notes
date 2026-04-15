#!/bin/bash

# 1. 安全进入目录。加上 || exit 1，防止 cd 失败时在错误目录下执行 git add .
cd ~/obsidian保存内容/Notes || { echo "进入目录失败"; exit 1; }

# 2. 先处理本地变动：暂存并提交
if [ -n "$(git status --porcelain)" ]; then
    echo "检测到本地变动，正在打包提交..."
    git add .
    git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "本地文件无变动。"
fi

# 3. 此时本地工作区绝对干净，执行拉取不会再报“未暂存”的错误
echo "拉取远端更新并合并..."
git pull origin main --rebase

# 4. 推送到远端仓库，并增加失败判断
echo "推送到远端仓库..."
if git push origin main; then
    echo "============= 同步成功：$(date '+%Y-%m-%d %H:%M:%S') ============="
else
    echo "============= 同步失败：请检查网络或 Git 冲突 ============="
fi
