# Linux
### ls
~~~
# 只显示目录
ls -d */

# 只显示文件（不含目录）
ls -p | grep -v /


~~~
## find
~~~
find . -name "ls"
find .  -type f -name 'cer*'
~~~ 


~~~
find . -type f -name "*.tmp" -delete
find . -type f -name "*.tmp" -exec rm {} \;
ls -lh
ls -lhR
ls -lht

~~~

## vim
|按键|功能|
|--|--|
|0	|行首|
|$	|行尾|
|gg	|文件开头|
|G	|文件末尾|
|w	|下一个单词开头|
|b	|上一个单词开头|
|o	|下一行（新建）|
|O	|上一行（新建）|
|dd	|删除整行|
|yy	|复制整行|
|p	|粘贴（光标后）|
|P	|粘贴（光标前）|
|u	|撤销|
|Ctrl+r	|重做|
|/pattern	|向下查找
|?pattern	|向上查找 
|n	|下一个匹配|
|N	|上一个匹配|
|:%s/old/new/	|替换当前行第一个|
|:%s/old/new/g	|替换当前行所有匹配项|
|:%s/old/new/	|替换当前行第一个|
|:%s/old/new/gc	|全局替换，逐个确认|
|:%s/old/new/gi	|全局替换，忽略大小写|


