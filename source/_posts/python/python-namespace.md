---
title: Python Namespace
id: python_namespace
category: python
date: 2019-01-24 15:47:17
tag: ['namespace', 'python']
---



如果缺少对 `python` 的 `namespace` 概念，在用 `python` 类的时候会对其对象属性值变化产生疑惑。

<!-- more -->

先看如下一段代码

```python
class T:
    a = 1

    def __init__(self):
        self.b = 2

    @classmethod
    def cls_foo(cls, num):
        cls.a = num
        cls.b = num

        
if __name__ == '__main__':
    obj1 = T()
    obj2 = T()
    print (obj1.a, obj1.b), (obj2.a, obj2.b)  # (1, 2), (1, 2)
    obj1.a = 11
    obj1.b = 22
    print (obj1.a, obj1.b), (obj2.a, obj2.b)  # (11, 22), (1, 2)
    obj1.cls_foo(33)
    print (obj1.a, obj1.b), (obj2.a, obj2.b)  # (11, 22), (33, 2)
```

上面代码的输出结果中，最后的 `obj2.a` 和 `obj2.a` 不同，且 `obj1` 和 `obj2` 也不同。这正是涉及了 `python` 的 `namespace`



## Namespace 

### 概念

namespace 是对于变量而言的，任何标量都属于一个 namespace ，而 namespace 有层级划分，主要分为 4 级，分别为

1. **local namespace** ，即对变量而言最近的一个闭包（**可以将一个层 <u>def 缩进</u> 视为一个 namespace**）
2. **enclosing function**，python 中可以在函数内定义函数，假设函数 ` foo` 内定义了函数 `bar` ，而函数 `bar` 内有变量 `spam`，则 `foo` 的范围就变量 `spam` 的 enclosing function （可以理解为变量所在 def 缩进层 的上级 def 缩进层）
3. **global namespace** ，一个模块就是一个global namespace
4. **builtin**，指的是 python 解释器的整个运行环境，函数 `abs` 就属于该 namespace



### 变量搜索

解释器运行一个python 语句 ( 含有变量的语句，如 `spam = 1` ) 时，会在 namespace 中根据变量名 ( `spam` ) 搜索指定变量进行操作。<u>搜索顺序</u>为 local -> enclosing function -> global -> builtin 。变量在 namespace 中逐级所搜，当查找到后就不再搜索下一个 namespace ，若所有 namespace 都查找不到则会丢出异常。



### 变量访问

根据变量的搜索过程可知道，在一个 namespace 中操作的标量不一定是属于该 namespace 的（即不是在该 namespace 中定义的），而是属于上级 namespace 的。这就引入了对变量访问操作时候的权限概念。

1. 一个 namespace 对所有定义在该namespace 的变量具有**读写权限**
2. 一个 namespace 对在其上级 namespace 内定义的变量具有**只读权限**
3. 一个 namespace 对其上级 namespace 定义的变量进行写入操作的时候会在本 namespace 内创建同名变量并进行写入操作（即创建同名副本进行读写）

这几条中的所谓的 “写操作” 对变量名进行写，变量名可视为 `C++` 里的指针，对变量名进行写操作即修改变量名所指向的地址（即赋值操作）。

根据第三条概念，就会有以下现象

```python
a = 1

def write_a():
    a = 2

def get_a():
    return a

if __name__ == '__main__':
    print get_a()  # 1
    write_a()
    print get_a()  # 1
```

在 `write_a` 中对 `a` 进行写操作，解释器会在 `write_a` 这个 namespace 内创建同名的 `a` 变量，而 `get_a` 这个函数则始终搜索到的是其上级 namespace 的 `a` 。



但是在python 中是不存在 “私有” 这个概念的，即以上三条的所谓的读写规则不是绝对的

> “Private” instance variables that cannot be accessed except from inside an object don’t exist in Python.

可以使用 `global` 和 `nonlocal` 语句对上级 namespace 的变量名进行写操作。

```python
a = 1

def write_a():
    a = 2

def get_a():
    return a

def write_global_a():
    global a
    a = 3

if __name__ == '__main__':
    print get_a()  # 1
    write_a()
    print get_a()  # 1
    write_global_a()
    print get_a()  # 3
```





> 参考： [9. Classes — Python 3.7.2 documentation](https://docs.python.org/3/tutorial/classes.html)

