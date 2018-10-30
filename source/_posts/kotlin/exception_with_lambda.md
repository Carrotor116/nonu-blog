---
title: "kotlin exception in android"
id: "kotlin_exception"
date: 2018-05-30
tags: ['kotlin']
---

kotlin 的 lambda 代码在 android 上出现的一次异常

<!--more-->
## kotlin 的解构
kotlin 中的类通过实现 `componentN` 方法，可以实现类的解构

```kotlin
data class Person(val name: String, val age: Int)
val person = Person("a", 1)
val (name, age) = person
```
一个解构声明会被编译成以下代码：
```kotlin
val name = person.component1()
val age = person.component2()
```
解构的参数如果有一个未使用，可以使用下划线代替 `(_, value)`



## lambda in kotlin
> lambda 表达式参数可以使用解构声明语法。 如果 lambda 表达式具有 Pair 类型（或 者 Map.Entry 或任何其他具有相应 componentN 函数的类型）的参数，那么可以通过将它们 放在括号中来引入多个新参数来取代单个新参数

```kotlin 
map.mapValues { entry -> "${entry.value}!" }
map.mapValues { (key, value) -> "$value!" }
```

> 注意声明两个参数和声明一个解构对来取代单个参数之间的区别：

```
{ a //-> …… } // 一个参数
{ a, b //-> …… } // 两个参数
{ (a, b) //-> …… } // 一个解构对
{ (a, b), c //-> …… } // 一个解构对以及其他参数
```



## 异常
看如下一段代码，其中第一种用法没有异常，而第二种用法在 android api 24 一下会出现异常
```kotlin
mapOf<String, String>().forEach { (t, u) ->
    t + 1
    u + 1
}
mapOf<String, String>().forEach { t, u ->   // may cause NoClassDefFoundError
    t + 1
    u + 1
}
```

> This happens due to a mixup of lambda signatures. The Java 8 Map.forEach() uses BiConsumer, which has a signature of (K, V) -> Unit. By contrast, the Kotlin version of Map.forEach() uses the signature (Map.Entry<K, V>) -> Unit. That means you have to destructure if you want two variables in Kotlin.
> 
> The Java 8 version of Map.forEach() wasn't added until API 24, so the code will crash on any older version of Android.

## Reference
* [解构声明 - Kotlin 语言中文站](https://www.kotlincn.net/docs/reference/multi-declarations.html)
* [android - java.lang.NoClassDefFoundError $$inlined$forEach$lambda$1 in Kotlin - Stack Overflow](https://stackoverflow.com/questions/42869086/java-lang-noclassdeffounderror-inlinedforeachlambda1-in-kotlin)
* [Kotlin Puzzler: Whose Line Is It Anyways?](https://blog.danlew.net/2017/03/16/kotlin-puzzler-whose-line-is-it-anyways/)
