#!/usr/bin/env node

var fs=require("fs");

if (process.argv.length != 3){
	console.log('usage: node tMath.js <filename>\ntranslate math in markdown file');
	process.exit();
}

var md=fs.readFileSync(process.argv[2],"utf-8");
var a = md;

var re=/\$([^\$\n]+?)\_([^\$\n]+?)\$/g;
while (a.search(re) > 0 ){
	a=a.replace(re, function(arg1, arg2, arg3,arg4,arg5){
		return "$"+arg2+"@@@@@@@"+arg3+"$";
	});
}
var re=/\$([^\$\n]+?)@@@@@@@([^\$\n]+?)\$/g;
while (a.search(re) > 0){
	a=a.replace(re, function(arg1, arg2, arg3,arg4,arg5){
		return "$"+arg2+"\\_"+arg3+"$";
	});
}

re=/\$([^\$\n]*?)\\\{([^\$\n]*?)\$/g;
while (a.search(re) > 0){
	a=a.replace(re, function(arg1, arg2, arg3,arg4,arg5){
		return "$"+arg2+"@@@@@@@"+arg3+"$";
	});
}
re=/\$([^\$\n]*?)@@@@@@@([^\$\n]*?)\$/g;
while (a.search(re) > 0){
	a=a.replace(re, function(arg1, arg2, arg3,arg4,arg5){
		return "$"+arg2+"\\\\\{"+arg3+"$";
	});
}

re=/\$([^\$\n]*?)\\\}([^\$\n]*?)\$/g;
while (a.search(re) > 0){
	a=a.replace(re, function(arg1, arg2, arg3,arg4,arg5){
		return "$"+arg2+"@@@@@@@"+arg3+"$";
	});
}
re=/\$([^\$\n]*?)@@@@@@@([^\$\n]*?)\$/g;
while (a.search(re) > 0){
	a=a.replace(re, function(arg1, arg2, arg3,arg4,arg5){
		return "$"+arg2+"\\\\\}"+arg3+"$";
	});
}

fs.writeFile(process.argv[2], a, function(err){
	if (err){
		console.log(err);
		process.exit();
	}
	console.log("translate math succeed")
})
