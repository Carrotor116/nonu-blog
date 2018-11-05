:: new a post in category paper
@echo off

if %1 == "" (
	echo usage: %0 ^<title^>
	echo new a post in category paper
	echo.

) else (
    hexo new post %1 --category paper
)

