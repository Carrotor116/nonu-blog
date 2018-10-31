var gulp = require('gulp');

//Plugins模块获取
var minifycss = require('gulp-clean-css');
var uglify = require('gulp-uglify');
var htmlmin = require('gulp-htmlmin');
var htmlclean = require('gulp-htmlclean');
var gutil = require('gulp-util')


// 压缩css文件
gulp.task('minify-css', function() {
  return gulp.src('./public/**/*.css')
  .pipe(minifycss())
//   .on('error', function(err) {
// gutil.log(gutil.colors.red('[Error]'), err.toString());
// })
  .pipe(gulp.dest('./public'));
});
// 压缩html文件
gulp.task('minify-html', function() {
  return gulp.src(['./public/**/*.html','!./public/**/politics/*'])
  .pipe(htmlclean())
  .pipe(htmlmin({
    removeComments: true,
    minifyJS: true,
    minifyCSS: true,
    minifyURLs: true,
  }))
  .pipe(gulp.dest('./public'))
});
// 压缩js文件
gulp.task('minify-js', function() {
    return gulp.src(['./public/**/.js','!./public/js/**/*min.js'])
        .pipe(uglify())
        .pipe(gulp.dest('./public'));
});
// 默认任务
gulp.task('default', [
  'minify-html','minify-css','minify-js'
]);