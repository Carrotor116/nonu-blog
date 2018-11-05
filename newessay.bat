:: new a post in category essay
@echo off

if %1 == "" (
	echo usage: %0 ^<title^>
	echo new a post in category essay
	echo.

) else (
    hexo new post %1 --category essay
)

