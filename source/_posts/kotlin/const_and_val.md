---
title: "kotlin 中的 const 和 val"
id: "const_and_val"
date: 2018-02-05
tags: ['kotlin']
---

kotlin 中 const 和 val 的关系。

<!--more-->


首先所有的 val 变量都会生成对应 getter 方法，var 变量会生成对应的 setter/getter 方法。

const 修饰符只能在 `top level of a file or inside an object`，且只能修饰基本类型和String类型，且需要立即初始化。
又 const 只能修饰 val，不能修饰 var。

即 `const val` 修饰对象会生成对应 java 代码的 `static final`




## 0x01 栗子
```kotlin
class KClass {

    val _val = 1                /*生成getter*/
    var _var = 1                /*生成getter和setter*/


    companion object {
        const val _conVal = 1   /*对应public final static */
        var _objVar = 1        
        val _objVal = 1        
        
        fun foo(){
            _objVar = 2         
        }
    }
}
```

* 注意在伴随类`companion object`中，`_objVar`不是public，是private。
在java代码中，能访问`KClass._conVal`，但不能`KClass._objVal`，IDE提示是`private`

* foo()方法在java中不能`KClass.foo()`调用，该方法属于`KClass$Companion.class`，非KClass，Kotlin中作为语法糖能够类似静态方法以`KClass.foo()`调用.



## 0x02 
上面的栗子中，foo 方法在 java 中该怎么调用？？？
可以这样 `KClass.Companion.foo();`
以下java代码可通过
```java
public class Manager {

    public static void main(String[] args) {

        KClass kClass = new KClass();

        int i = KClass._conVal;
        
        KClass.Companion.foo();
        KClass.Companion.get_objVal();
        KClass.Companion.set_objVar(1);
    }
}
```



## 0x02 kotlin to java
1. Menu > Tools > Kotlin > Show Kotlin Bytecode
2. Click on the Decompile button
3. Copy the java code



## Reference
> 参考：[Kotlin---------------const详解](http://www.mamicode.com/info-detail-1929175.html)
> 参考：[How to convert a kotlin source file to a java source file](https://stackoverflow.com/questions/34957430/how-to-convert-a-kotlin-source-file-to-a-java-source-file)