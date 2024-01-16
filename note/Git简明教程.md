# Git简明教程

## 是什么?有什么用?
Git是当前最主流的**分布式版本控制系统**(VCS),这可以帮你保存文件的修改记录并用版本号区分,起到**恢复和保护**的作用

## Git的使用

### 创建新仓库
您可以使用一个已经存在的目录作为Git仓库或创建一个空目录。
使用您当前目录作为Git仓库，我们只需使它初始化。
`git init`

使用我们指定的目录作为Git仓库
`git init newrepo`

### 检出仓库
执行如下命令以创建一个本地仓库的克隆版本:
`git clone /path/to/repository`

如果是远端服务器上的仓库,你的命令会是这个样子:
`git clone ssh://example.com/~/www/project.git`

### 工作流
你的本地仓库由 git 维护的三棵“树”组成。第一个是你的 **工作目录**，它持有实际文件；第二个是 **暂存区（Index）**，它像个缓存区域，临时保存你的改动；最后是 **HEAD**，它指向你最后一次提交的结果。

### 添加和提交
添加并不是提交代码到远程Git库，Git也并不会你修改了代码它自动帮你保存你修改的每一个过程。你修改了很多文件，但未必所有的修改，最终打算提交上去，那么哪些是你打算提交的，你可以添加进来待会提交，叫做缓存改动。很简单，比如本地电脑上我有整个项目完整的东东，甚至包含了账号密码的一些文件，但是我只是ADD除账号密码之外的文件，并不缓存账号密码文件的改动。不被ADD它就不会参与后续的操作。通常我都会直接全部缓存，它会自动寻找所有有改动的文件，而不需要提交的文件放在忽略的文件夹中。

你可以提出更改（把它们添加到暂存区），使用如下命令:
`git add <filename>`
`git add*`

这是 git 基本工作流程的第一步；使用如下命令以实际提交改动(提交版本):
`git commit -m"代码提交信息"`
如果您不使用-m，会出现编辑器来让你写自己的注释信息。

可以输入任意内容，当然最好是有意义的，这样你就能从历史记录里方便地找到改动记录。

当我们修改了很多文件，而不想每一个都add，想commit自动来提交本地修改，我们可以使用-a标识。
`git commit -a -m "Changed some files"`
git commit 命令的-a选项可将所有被修改或者已删除的且已经被git管理的文档提交到仓库中。（all）

**千万注意，-a不会造成新文件被提交，只能修改。**

现在，你的改动已经提交到了 HEAD，但是还没到你的远端仓库。

总结：先add后commit

### 推送改动
你的改动现在已经在本地仓库的 HEAD 中了。执行如下命令以将这些改动提交到远端仓库：
`git push origin master`
可以把 master 换成你想要推送的任何分支。

第一次推送master分支时，加上-u参数。Git不但会把本地的`master`分支内容推送的远程新的`master`分支，还会把本地的`master`分支和远程的`master`分支关联起来，在以后的推送或者拉取时就可以简化命令。

`git push -u origin master`

推送到服务器

`git push ssh://example.com/~/www/project.git`

如果你还没有克隆现有仓库，并欲将你的仓库连接到某个远程服务器，你可以使用如下命令添加：
`git remote add origin <server>`
如此你就能够将你的改动推送到所添加的服务器上去了。

### 删除
从资源库中删除文件：
`git rm file`

并且：

`git commit`



删除远程库：(其实是解除了本地和远程的绑定关系，并不是物理上删除了远程库。远程库本身并没有任何改动。要真正删除远程库，需要登录到GitHub，在后台页面找到删除按钮再删除。)

`git remote rm <name>`

建议先查看远程库信息：

`git remote -v`

### 查看日志和状态
可以告诉我们历史记录
`git log`

记录每一次命令
`git reflog`

查看状态

`git status`

### 工作区和暂存区
#### 工作区（Working Directory）
你在电脑里能看到的目录

#### 版本库（Repository）
工作区中的隐藏目录.git,不算工作区，而是Git的版本库
其中包含
+ **暂存区stage（index）**

+ **第一个分支master**

+ **指向master的一个指针HEAD**

  需要提交的文件修改通通放到暂存区，然后，一次性提交暂存区的所有修改。

### 分支
分支是用来将特性开发绝缘开来的。在你创建仓库的时候，master 是“默认的”分支。在其他分支上进行开发，完成后再将它们合并到主分支上。

仅仅创造一个叫test的分支
`git branch test`
创建一个叫做“feature_x”的分支，并切换过去：（-b表示创建并切换）
`git checkout -b feature_x`

查看当前分支:(当前分支前会标一个*号)

`git branch`

切换回主分支：
`git checkout master`

最新版本的switch切换到新的dev分支：

`git switch -c dev`

switch切换到已有的master分支：

`git switch master`

再把新建的分支删掉：//-d标识
`git branch -d feature_x`

除非你将分支推送到远端仓库，不然该分支就是 不为他人所见的：
`git push origin <branch>`

#### bug分支
修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除；

当手头工作没有完成时，先把工作现场stash一下:
`git stash`
然后去修复bug，修复后，再:
`git stash pop`
回到工作现场

在master分支上修复的bug，想要合并到当前dev分支，可以用:
`git cherry-pick <commit>`
把bug提交的修改“复制”到当前分支，避免重复劳动。

#### feature分支
开发新feature，新建新branch

如果要丢弃一个没有被合并过的分支，可以:
`git branch -D <name>`

#### rebase
rebase操作可以把本地未push的分叉提交历史整理成直线；
rebase的目的是使得我们在查看历史提交的变化时更容易，因为分叉的提交需要三方对比。
`git rebase`

### 取回更新与合并
要更新你的本地仓库至最新改动，执行：
`git pull`
以在你的工作目录中 获取（fetch） 并 合并（merge） 远端的改动。（当前分支自动与唯一一个追踪分支进行合并）

从非默认位置更新到指定url
`git pull http://git.example.com/project.git`

要合并其他分支到你的当前分支（例如 master），执行：
`git merge <branch>`
合并分支时，加上**--no-ff**参数就可以用普通模式合并，合并后的历史有分支，能看出来曾经做过合并，而fast forward合并就看不出来曾经做过合并。

在这两种情况下，git 都会尝试去自动合并改动。遗憾的是，这可能并非每次都成功，并可能出现冲突（conflicts）。 这时候就需要你修改这些文件来手动合并这些冲突（conflicts）。改完之后，你需要执行如下命令以将它们标记为合并成功：
`git add<filename>`

在合并改动之前，你可以使用如下命令预览差异：
`git diff <source_branch> <target_branch>`

查看工作区和版本库里面最新版本的区别：

`git diff HEAD -- readme.txt`

### 解决冲突
当Git无法自动合并分支时，就必须首先解决冲突。解决冲突后，再提交，合并完成。
解决冲突就是把Git合并失败的文件手动编辑为我们希望的内容，再提交。
查看分支合并图:
`git log --graph`

### 标签
为软件发布创建标签是推荐的。这个概念早已存在，在 SVN 中也有：
`git tag <tagname>`

你可以执行如下命令创建一个叫做 1.0.0 的标签：
`git tag 1.0.0 1b2e1d63ff`
1b2e1d63ff 是你想要标记的提交 ID 的前 10 位字符。

可以使用下列命令查看所有标签：
`git tag`

#### 标签推送
如果要推送某个标签到远程，使用命令
`git push origin <tagname>`

一次性推送全部尚未推送到远程的本地标签：
`git push origin --tags`

#### 删除已推送到远程的标签
先从本地删除:
`git tag -d <tagname>`

从远程删除:
`git push origin :refs/tags/<tagname>`


### 替换本地改动
假如你操作失误（当然，这最好永远不要发生），你可以使用如下命令替换(撤销)掉本地改动：
`git checkout -- <filename>`
此命令会使用 HEAD 中的最新内容替换掉你的工作目录中的文件。已添加到暂存区的改动以及新文件都不会受到影响。

把暂存区的修改撤销掉（unstage），重新放回工作区

`git reset HEAD <FILE>`

假如你想丢弃你在本地的所有改动与提交，可以到服务器上获取最新的版本历史，并将你本地主分支指向它：
`git fetch origin`
`git reset --hard origin/master`

### 实用小贴士
内建的图形化 git：
`gitk`

彩色的 git 输出：
`git config color.ui true`

显示历史记录时，每个提交的信息只显示一行：
`git config format.pretty oneline`

交互式添加文件到暂存区：
`git add -i`


### Git Cheat Sheet
https://liaoxuefeng.gitee.io/resource.liaoxuefeng.com/git/git-cheat-sheet.pdf


