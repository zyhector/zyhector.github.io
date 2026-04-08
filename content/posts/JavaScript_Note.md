---
title: "JavaScript笔记"
subtitle: ""
date: 2023-07-07T22:55:48+08:00
draft: false
toc:
    enable: true
weight: false
categories: ["问学书斋"]
tags: ["JavaScipt", "编程语言", "笔记"]
---

## 写在前面

[照着这个视频学的JavaScript](https://www.bilibili.com/video/BV1ux411d75J)

刷的很快，一下午跳着看完了16小时的内容，毕竟ushio之前学过cpp和python，基本上就了解一下语法，很多概念都不需要重复去啃了。

这里是简单的记一记课堂笔记，万一忘了还得回来看看。

## 浏览器执行JS简介

浏览器分成两部分，渲染引擎与JS引擎

* 渲染引擎：用来解析HTML和CSS，俗称“内核”，比如chrome的blink，老版本的webkit
* JS引擎：JS解释器，解析和执行JS代码，如chrome中的V8，

## JS的组成

* ECMAScript: JavaScript语法

ECMA国际规定了JS的变成语法和基础核心知识，是所有浏览器厂商共同遵守的一套JS语法工业标准。

* DOM: 页面文档对象模型

Document Object Model，是W3C组织推荐的处理可扩展标记语言的**标准编程接口**。通过DOM提供的接口可以对页面上的各种元素进行操作。

* BOM: 浏览器对象模型

Browser Object Model，浏览器对象模型，提供了独立于内容的、可以与浏览器窗口进行互动的对象结构。通过BOM可以操作浏览器窗口，比如弹出框、控制浏览器跳转、获取分辨率等。

## JS初体验

JS有三种书写位置，分为行内、内嵌和外部。

在HTML中推荐使用双引号，在JS中推荐使用双引号

内嵌式是最常用的

```JavaScript
<!DOCTYPE html>

<head>
    <title>Document</title>
    <!-- 内嵌式的js，用script包裹 -->
    <script>
        alert("你好哇")
    </script>
    <!-- 外部js，通过src引入执行 -->
    <script src="my.js"></script>
</head>
<body>
    <!-- 行内式的js，直接写到元素的内部 -->
    <input type="button", value="huahua" onclick="alert('kawaii')">
</body>
</html>
```

## JS的注释

单行注释 ```//```

多行注释 ```/* */```

## JS的输入输出语句

|方法|说明|归属|
|--|--|--|
|alert(msg)|浏览器弹出警告框|浏览器|
|console.log(msg)|浏览器控制台打印输出信息|浏览器|
|prompt(info)|浏览器弹出输入框，用户可以输入|浏览器|

## 变量

声明 ```var i, j, k;```、

var是variable，JS的关键字用于声明变量，由计算机自动分配内存空间。

（JS是弱变量类型，不用声明类型）

赋值 ```i = 1;```

初始化 ```var i = 1;```

也可以不声明就直接赋值使用，但是变量会变成全局变量，不提倡使用。

命名规范

* AZaz09_$ 可以组成变量
* 严格区分大小写
* 不能以数字开头
* 小驼峰命名

## 数据类型

简单数据类型：Number, String, Boolean, Undefined, Null

|简单数据类型|说明|默认值|
|-|-|-|
|Number|整型、浮点|0|
|Boolean|true, false|false|
|String|字符串|""|
|Underfined|声明但是未赋值的变量|undefined|
|Null|val a = null;|null|

以0开头的是8进制（0o），0x、0b等等。

```JavaScript
Number.MAX_VALUE = 1.797e+308;
Number.MIN_VALUE = 5e-324;
Infinity;
-Infinity;
NaN;
```

```JavaScript
isNaN() // 判断是不是数字
数字 -> false
NaN -> true
```

```JavaScript
str.length // 字符串长度
str = str1 + var // 字符串可以和其他类型拼接称字符串
undefined + str = str
undefined + Numbar = NaN
```

获取数据类型
```JavaScript
console.log(typeof x);
```

数据类型转换

表单、prompt默认是字符串类型的

```JavaScript
// Number -> String
num.toString();
String(num);
num + ''; // 隐式转换，比较常用哦
```

```JavaScript
// String -> Number
parseInt(string);
parseFloat(string)
Number()
str - 0; // 减号的隐式转换不一样
str * 1;
str / 1;
```

```JavaScript
// -> Number
Boolean()
```

## 标识符、关键字、保留字

标识（zhi4）符：开发人员为变量、属性、函数、参数取的名字。

关键字：JS本身已经使用的字，不能再当变量名、方法名。

保留字：预留的关键字，现在还不是关键字，但可能会变成关键字。

## 运算符

### 算术运算符

```+ - * / %```

```++ --```一样是变量放前面放后面都行，规律同C++

### 比较运算符

```> < >= <= == != === !==```

```===```值和数据类型都要相同

### 逻辑运算符

```&& || !```

有短路，返回值是最后的一个子式。

### 赋值运算符

```= += -= *= /= %=```

### 运算符优先级

|优先级|运算符|顺序|
|-|-|-|
|1|小括号|()|
|2|一元运算符|++ -- !|
|3|算数运算符|先*/% 后+-|
|4|关系运算符|> >= < <=|
|5|相等运算符|== != === !==|
|6|逻辑运算符|先&& 后\|\||
|7|赋值运算符|=|
|8|逗号运算符|,|

## 分支

三目：```? :``` 规则同C++

```JS
if (condition) {

}
else if () {

}
else {

}
```

```JS
switch() {
    case value:
        dosomething;
        break;
    default: // 都没有匹配上
        dosomething;
}
```

## 循环

```JS
for (var i = 1; i <= 100; i++) {
    xxxx;
}
while () {
    xxxx;
}
do {
    xxxx
} while ()
```

## 数组

```JS
var arr = new Array();
var arr = ['a', 'b', 'c', 'd']; // 数组字面量创建数组
```
从零开始，中括号，下标访问。

新增元素：

直接修改```arr.length```

用新下标直接赋值。

## 函数

```JS
function fun(a, b, c) {
    // 如果实参个数多余形参个数，只取形参个数的实参
    // 如果实参个数小于形参个数，会变成undefined
    return xxx; // 没有return，返回值为undefined
}
var fun = function() {
    // 哇哦，是匿名函数欸
}
```

所有函数内置一个arguments对象，其中存储了**传递过来的所有实参**

有对应的length方法，可以通过下标遍历arguments

## 解析执行

js引擎运行js分为两步：预解析、代码执行

预解析：js引擎把js里面所有的var和function提升到当前作用域的最前面

代码执行：按书写顺序从上往下执行

变量提升只提升变量声明，不提升赋值操作

## 对象

对象是一组无序的相关属性和方法的集合

```JS
var obj = {
    a: 1,
    b: 2,
    fun: function() { // 方法的冒号后面要跟一个匿名函数
        xxxxxx;
    }
}
// 调用
obj.a
obj['b']
obj.fun()
```
另一种方法
```JS
var obj = new Object();
obj.a = 1;
obj.b = 2;
obj.fun = function() {
    xxxx;
}
```
构造函数
```JS
function constructFun() {
    this.a = 1;
    this.b = 2;
}
var obj = new constructFun();
```
遍历对象
```JS
for (var k in obj) { // for in 一般用 k key
    console.log(k); // 打印属性名
    console.log(obj[k]); // 打印属性值
}
```

## 简单数据类型与复杂数据类型

简单数据类型存放到栈里，
复杂数据类型存放到堆里。

简单数据类型传参的时候复制了一份，而复杂数据类型只传了指针（但也不是指针，总之就是没有copy）

## 写在后面

呃，好快的学...

其实是因为很迫切的想学怎么ui编程，而不是再面对黑乎乎的框框去写代码。所以听了YYY的建议打算学Vue。所以就开始看JS。所以一会会之后就会去学Vue了，应该也会有一篇新的blog的！