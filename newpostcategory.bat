:: new a post in special category 
@echo off

if "%1" == "" goto usage
if "%2" == "" goto usage 

hexo new post %2 --category %1


: usage
	echo usage: %0 ^<category^> ^<title^>
	echo new a post in special category
	echo.