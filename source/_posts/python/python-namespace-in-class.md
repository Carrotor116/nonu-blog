---
title: Python Namespace in Class
id: python_namespace_in_class
category: python
date: 2019-01-24 15:48:32
tag: ['namespace', 'python', 'class']
---



## Class

前文 ([Python Namespace](python_namespace)) 说了，可以将一层 **def 缩进** 视为一个 namespace ，而类在 python 中使用 `class` 定义的，一个 **class 缩进** 也可以视为一个 namespace

```python
class ClassName
    <statement-1>
    ...
    <statement-1>
```



在 python 中所有数据结构都是对象，即都是**对象实例**。

1. **<u>类</u>**也是一个对象实例，在定义一个类后，python 生成一个**对象实例**

2. **<u>类实例</u>**是也是对象实例，在实例化一个类后，python 生成一个**对象实例**

<u>一个类可以视为一个 namespace ，一个类实例也可以视为一个 namespace</u> ，其关系是类的 namespace 是类实例 namespace 的上级。

<!-- more --->

### 类的变量搜索

可以将类视为一个 namespace ，namespace 的变量搜索规则同样在类的变量中使用，及先搜索本 class 的namespace ，不存在再搜父类的 namespace 。此外，`python` 的类支持多继承，搜索变量时按照 **从左到右** 的顺序搜索父类的namespace。 **实际上类的变量和方法的搜索顺序均是按照 `__mro__` 列表顺序进行搜索解析**，于下文介绍。

> For most purposes, in the simplest cases, you can think of the search for attributes inherited from a parent class as depth-first, left-to-right, not searching twice in the same class where there is an overlap in the hierarchy. Thus, if an attribute is not found in DerivedClassName, it is searched for in Base1, then (recursively) in the base classes of Base1, and if it was not found there, it was searched for in Base2, and so on.



可以将类实例视为一个 namespace，对类实例的变量搜索，其顺序是先搜索 类实例的 namespace ，再搜索其类 的 namespace 。



### 类的变量访问	

与 namespace 的变量访问规则相同。类在进行变量写操作时，如果类的 namespace 中不存在，则会在 类的 namespace 中创建变量。

而由于 python 不存在私有的概念，类实例也可以访问操作类的变量，通过使用 `@classmethod` 修饰符实现。 `@classmethod` 修饰器定义的方法，第一个参数是 `cls` ，而普通方法的第一个参数是 `self` ，这为对不同 namespace 内的变量进行访问操作提供了可能。即类实例可以通过普通方法对类实例本身的 namespace 内的变量进行访问操作；类实例通过 `@classmethod` 方法对类的 namespace 内变量进行访问操作。

（ `@classmethod` 定义的方法中第一个参数 `cls` 表示本类，实际上 `cls.par` 的操作均可以替换为 `ClassName.par` 替换，不过使用 `cls` 可以在类继承中实现动态调用 ）



### 类变量操作栗子

#### 栗子1

这样结合前文中 `namespace` 的<u>读写规则</u> 和 `变量` 的<u>搜索顺序</u>，再看前文的第一个 demo

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

其中 `@classmethod` 修饰器定义的方法，第一个参数是 `cls` ，而普通方法的第一个参数是 `self` ，这为对不同 namespace 内的变量进行访问操作提供了可能。

（ 实际上这个 demo 中的 `cls.a` 可以替换为 `T.a` ，不过使用 `cls` 可以在类继承中实现动态调用 ）

现在具体分析下这个 demo 。

1. 在类的定义中，定义了变量`a = 1` ，该变量属于 类T 的 namespace 
2. 然后在`__init__` 函数中定义了 `self.b` ，该变量属于 类实例的 namespace ( 虽然在创建类实例之前，不存在类实例 ) ，不属于 类T ，即不存在 `T.b` 
3. 接下来定义 `cls_foo` 函数。具有 `@classmethod` 修饰，则第一个参数表示类，故 `cls.a` 是对类这个namespace 内的 `a` 进行访问操作，而 `cls.b` 同样是对类的 namespace 中的 `b` 进行操作。（ **注意** `cls.b = num` 属于赋值操作，若该变量不属于本 namespace ，则会在本 namespace 中创建同名的变量，再进行赋值操作 ）
4. `if_main` 的第四句 `obj1.a = 1` 会在 `obj1` 的namespace 内创建 `a` 变量，并且赋值。而 `obj2` 的 namespace 中始终不存在 `a` 变量
5. `if_main` 的第七句，调用 `obj1.cls_foo(33)` ，是对 类T 的namespace 中标量进行访问操作，这里会由于  `cls.b = num` 而导致在 类T 的 namespace 中创建变量 `b` 
6. `if_main` 的第八句，读 `obj1` 和 `obj2` 的 `a` 和 `b` 。`obj1` 的namespace 中存在这两个变量，分别为 11 和 22 ，而 `obj2` 的namespace 中始终不存在 `a` 标量，故 `obj2.a` 访问的是 类T 的 namespace 中的变量 `a` ，即 `T.a` 。所以最后输出的是 `(11, 22), (33, 2)` 



#### 栗子2

再看另一个改自 python 官方文档的例子

```python 
class Dog:
    tricks = []             # mistaken use of a class variable
    age = -1
    def __init__(self, name):
        self.name = name
    def add_trick(self, trick):
        self.tricks.append(trick)
    def set_age(self, age):
        self.age = age

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.add_trick('roll over')
>>> e.add_trick('play dead')
>>> d.tricks                # unexpectedly shared by all dogs
['roll over', 'play dead']
>>> d.set_age(1)
>>> e.set_age(2)
>>> d.age, e.age
(1, 2)   # the age of obj is independent
```

上面的 demo 中 `tricks` 出现了共享的现象，而 `age` 却是独立的。

因为类对象实例 `d` 和 `e` 的 `namespace` 中均不存在 `tricks` ，所以 `self.tricks` 实际上使用的是类对象 `Dog` 的 `namespace` 中的 `tricks` ，所以语句 `self.tricks.append` 是调用同一个 `list` 对象的 `append` 方法。

在执行 `self.age = age` 的时候，原本只存在 `Dog.age` ，会自动在 类实例 的 `namespace` 内生成同名的 `age` 变量并对其复制，故此时实际上存在三个同名的 `age` 变量，即 `Dog.age`，`d.age`，`e.age`。



### 类的方法访问

类的普通方法带有 `self` 参数，需要使用类实例对象进行调用；使用 `@classmethod` 修饰的方法带有 `cls` 参数，可以由类对象直接调用。

类调用其父类可以使用 `super` 关键字，具体方式如下

```python 
class A(object):  
    # A should extend a class otherwise will raise error when use super
    def foo(self):
        print 'call A.foo'


class B(A):
    def foo(self):
        super(B, self).foo()  # this is syntax in python 2,
                              # and can replace with super().foo() in python 3
        print 'call B.foo'


if __name__ == '__main__':
    obj = B()
    obj.foo()
```

output : 

```
call A.foo
call B.foo
```

调用 `super` 查找方法时，按照 类的 `__mro__` （“Method Resolution Order”）属性列表进行依次对父类进行查找。

在python 2.1 及以前只有 `Old-style Class`， `__mro__` 采用深度优先算法构造，在 python 2.2 以及之后`Old-style Class` 依然使用深度优先算法构造 ，`New-style Class` 使用 C3 线性化算法构造，python 3 则只有 `New-style Class` ，使用 C3 线性化算法构造。 [C3 线性化算法与 MRO - Kaiyuan's Blog | May the force be with me](http://kaiyuan.me/2016/04/27/C3_linearization/) 一文对此介绍很清楚。





> 参考： [9. Classes — Python 3.7.2 documentation](https://docs.python.org/3/tutorial/classes.html)
>
> 参考：[8.7 调用父类方法 — python3-cookbook 3.0.0 文档](https://python3-cookbook.readthedocs.io/zh_CN/latest/c08/p07_calling_method_on_parent_class.html)
>
> 参考：[C3 线性化算法与 MRO - Kaiyuan's Blog | May the force be with me](http://kaiyuan.me/2016/04/27/C3_linearization/)