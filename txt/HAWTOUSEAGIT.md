# gitの使い方
**事故防止のためmainに直接push禁止**
featureブランチ下で新たな機能を実装後、mainに反映する。
xxxでそのブランチで実装する機能を示す。
## ローカルを更新
git switch main
git pull origin main

## ブランチに移動
- 新規機能
    - git switch -c feature/xxx
- 既存機能
    - git switch feature/xxx

feature/xxxで実装する

## 保存
git add .
git commit -m "やったことを書く"


## githubへ送信
git branchで
*feature/xxxになっているか確認後
- 初回
    - git push -u origin feature/xxx
- 既存ブランチ
    - git push

## GitHub上で「Merge」

## ローカルを更新
git switch main
git pull origin main

# 便利
- git status
- git log
- git fetch