# git commend

## 1 Git global setup

~~~
git config --global user.name "Kelvin Qin"
git config --global user.email "kelvinqin@globalsources.com"
~~~

## 2 Create a new repository

~~~
git clone http://gsol-gitlab.szn.globalsources.com/isearch/gsol-isearch-service-api.git
cd gsol-isearch-service-api
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
~~~

## 3 Push an existing folder

~~~
cd existing_folder
git init
git remote add origin http://gsol-gitlab.szn.globalsources.com/isearch/gsol-isearch-service-api.git
git add .
git commit -m "Initial commit"
git push -u origin master
~~~

## 4 Push an existing Git repository

~~~
cd existing_repo
git remote rename origin old-origin
git remote add origin http://gsol-gitlab.szn.globalsources.com/isearch/gsol-isearch-service-api.git
git push -u origin --all
git push -u origin --tags
~~~

## 5 git create/delete branch
~~~
 // create branch & commit
 git checkout -b feature/kelvin/s139/master
 git push -u origin feature/kelvin/s139/master
 
 // delete local/remote branch & commit
 git -D feature/kelvin/s139/master
 git push origin --delete  feature/kelvin/s139/master
 
~~~

## 6 git merge
~~~
 git merge --no-ff feature/kelvin/s139/master
~~~
