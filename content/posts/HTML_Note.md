---
title: "HTML笔记"
subtitle: ""
date: 2023-07-10T11:34:39+08:00
draft: false
toc:
    enable: true
weight: false
categories: ["问学书斋"]
tags: ["HTML", "笔记"]
---

## 写在前面

[这次是跟着这个学的](https://www.bilibili.com/video/BV1vs411M7aT)

去看了一眼Vue的官方教程，但是感觉还有好多啃不下来的HTML的知识，所以打道回府，学一下HTML。

## HTML

HyperText Markup Language

这玩意是一个标记语言，**不是编程语言**。（但是不重要啦）

## 元素

```HTML
<p> Hello World </p>
前面的<p>叫 Opening tag
中间的称为Content
后面的叫 Closing tag
整一个作为一个HTML Element
```

## 文档结构

```HTML
<!DOCTYPE html> 解释文档的类型
<html>
    <head>
        这一部分内容不会呈现在网页上
        可以放一些比如搜索引擎关键字啊
        网页的标题啊等等的信息
        <title>This is my title</title>
    </head>
    <body>
        真正渲染的部分
        <h1></h1>
        <p>paragraph</p>
    </body>

</html>
```

## 块级元素和内联元素

块级元素：
* 在页面以块的形式展现
* 出现在新的一行
* 占全部宽度

```<div> <h1>-<h6> <p>```

内联元素：
* 通常在块级元素内
* 不会导致文本换行
* 只占必要的部分宽度

```<a> <img> <em> <strong>```

## 各种示例

```HTML
<strong> 加粗 </strong>
<em> 斜体 </em>
<a href="url" target="_blank"> attribute </a>
<!-- 用target可以在新页面打开链接哟 -->
<br> 换行
<hr> 水平线

<!-- Lists 列表 -->
<ul>
    <li> * 是这种样式的列表 </li>
</ul>
<ol>
    <li> 1. 这个是有序的列表 </li>
</ol>

<!-- Table 表格 -->
<table>
    <thead>
        <tr>
            <th>First Name</th>
            <th>Second Name</th>
        </tr>
    </thead>
    <body>
        <tr>
            <td>data</td>
        </tr>
    </body>
</table>
<!-- 
tr: table roll
th: table head
td: table data 
-->


<!-- Form 表单 -->
<form action="form.js">
    <div>
        <label>First Name</label>
        <input type="text" name="firstname" placeholder="Enter first name">
    </div>
    <div>
        <label>Email</label>
        <input type="email" name="email" placeholder="Enter email">
    </div>
    <input type="submit" name="submit" value="Submit">
    <div style="margin-top:1000px"></div>
</form>

<!-- Button -->
<button>Button</button>

<!-- Image -->
<img src="xxx" alt="替换的文字">

<!-- Quotation -->
<blockquote></blockquote>

<abbr title="悬浮显示的注释一类的（？）">xxxxxx</abbr>


```

## 各种学到的tips

在vsc按```alt+shift+up/down```可以快速复制当前行（好像只有html里能用？）

lorem10 再按tab可以自动生成10个占位词

