---
title: Name Mangling in Python Class
id: name_mangling_in_python_class
category: python
date: 2019-01-24 15:48:46
tags: ['private attributes', 'python virtual func']
---



`name mangling` 在官方文档中的描述

> Any identifier of the form `__spam` (at least two leading underscores, at most one trailing underscore) is **textually** replaced with `_classname__spam`, where `classname` is the current class name with leading underscore(s) stripped.

即**在类定义中**，以两个及其以上下划线开头且以一个以内下划线结束的命名的任意标识符 `_spam`，都会被逐字替换后加上其类名，变成`_classname_spam` 。



<!-- more --> 

## 使用技巧

### Trick 1: 私有变量的实现

> “Private” instance variables that cannot be accessed except from inside an object don’t exist in Python.

python 中不存在 “私有” 的概念，执行语句是按照 namespace 的变量搜索顺序查找变量并对变量进行访问操作。结合 `name mangling` 机制，可以实现 “伪私有”。 如下

```python 
class A:
    _i = 1
    __j = 2

if __name__ == '__main__':
    o = A()
    print(o._i)    # 1
    print(o._A__j) # 2
    print(o.__j)   # AttributeError: 'A' object has no attribute '__j'
```

即通过 `name mangling` 的自动替换，类中将不存在任意以双下划线开头的属性，使得上面 demo 中 `o` 不能访问 `__j` ，因为实际上，在 `python` 的 `namespace` 中不存在 `__j` ，而是 `_A__j` 。

注意 `name mangling` 之作用于类定义，故在 `if_main` 中的编码 `print(o.__j)` 不会被自动替换。



### Trick 2: 阻止类方法覆盖

`python` 官方文档中也有同样的例子

```python
class A:
    def bar(self):
        print 'call A.bar'
        self.bar()
        self.__foo()  # this will be replace with self._A_foo automatically

    def foo(self):
        print 'call A.foo'

    __foo = foo  # this will be replace with _A__foo = foo automatically


class B(A):
    def bar(self):
        print 'call B.bar'

    def foo(self):
        print 'call B.foo'

    __foo = foo  # this will be replace with _B__foo = foo

    
if __name__ == '__main__':
    b = B()
    A.bar(b)
```

output : 

```
call A.bar
call B.bar
call A.foo
```

`A.bar` 函数中的 `self.bar()` 是动态调用，具有 `C++` 中虚函数的效果，而 `__foo` 属性通过 `name mangling` 机制的自动替换，可以实现对 `A.foo` 的硬编码，防止继承覆写函数的动态调用。





> 参考： [9. Classes — Python 3.7.2 documentation](https://docs.python.org/3/tutorial/classes.html)