---
title: "线性代数笔记"
subtitle: ""
date: 2023-07-18T21:26:17+08:00
draft: true
toc:
    enable: true
weight: false
categories: ["问学书斋"]
tags: ["线性代数", "机器学习"]
---

## 写在前面

日，你交的线代课没有教矩阵求导，导致ushio被机器学习里的线代基础给创死了。

赶紧补习一下。

# 向量变元的实值标量函数

$$ f(\mathbf{x}), \mathbf{x} = [x_1, x_2, \dots, x_n]^T $$

$$ \dfrac{\partial f(\mathbf{x})}{\partial \mathbf{x}} = [\dfrac{\partial f}{\partial x_1}, \dfrac{\partial f}{\partial x_2}, \dots, \dfrac{\partial f}{\partial x_n}] $$

## 基本法则

这里都和标量函数差不多。

### 常数求导

$$ \dfrac{\partial c}{\partial \mathbf{x}} = \mathbf{0}_{n \times 1} $$

### 线性法则

$$ \dfrac{\partial c_1 f(\mathbf{x}) + c_2 g(\mathbf{x})}{\partial \mathbf{x}} = c_1 \dfrac{\partial f(\mathbf{x})}{\partial \mathbf{x}} + c_2 \dfrac{\partial g(\mathbf{x})}{\partial \mathbf{x}} $$

### 乘积法则

$$ \dfrac{\partial f(\mathbf{x}) g(\mathbf{x})}{\partial \mathbf{x}} = \dfrac{\partial f(\mathbf{x})}{\partial \mathbf{x}} g(\mathbf{x}) + f(\mathbf{x}) \dfrac{\partial g(\mathbf{x})}{\partial \mathbf{x}} $$

### 商法则

$$ \dfrac{\partial [\frac{f(\mathbf{x})}{g(\mathbf{x})}]}{\partial \mathbf{x}} = \dfrac{1}{g^2(\mathbf{x})} [\dfrac{\partial f(\mathbf{x})}{\partial \mathbf{x}} g(\mathbf{x}) + f(\mathbf{x}) \dfrac{\partial g(\mathbf{x})}{\partial \mathbf{x}}] $$

## 几个柿子

$$ \dfrac{\partial (\mathbf{x}^T \mathbf{a}) }{\partial \mathbf{x}} = \dfrac{\partial (\mathbf{a}^T \mathbf{x}) }{\partial \mathbf{x}} = \mathbf{a} $$

$$ \dfrac{\partial (\mathbf{x}^T \mathbf{x})}{\partial \mathbf{x}} = 2 \mathbf{x} $$

$$ A_{n\times n}, \dfrac{\partial (\mathbf{x}^T A \mathbf{x})}{\partial \mathbf{x}} = A \mathbf{x} + A^T \mathbf{x}$$

$$ \dfrac{\partial( \mathbf{a}^T \mathbf{x} \mathbf{x}^T \mathbf{b})}{\partial \mathbf{x}} = \dfrac{\partial( (\mathbf{a}^T \mathbf{x}) (\mathbf{x}^T \mathbf{b}))}{\partial \mathbf{x}} = \dfrac{\partial( \mathbf{x}^T \mathbf{a} \mathbf{b}^T \mathbf{x})}{\partial \mathbf{x}} = \mathbf{ab}^T\mathbf{x} + \mathbf{ba}^T\mathbf{x}$$

# 矩阵变元的实值标量函数

> 还没学完，先更一半