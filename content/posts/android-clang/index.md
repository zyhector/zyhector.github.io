---
title: "C++编译安卓原生二进制的一个简单方法"
subtitle: ""
date: 2025-03-11T13:39:23+08:00
toc:
    enable: true
weight: false
categories: ["问学书斋"]
tags: ["Android", "LLVM", "clang", "编译", "配环境", "C++"]
---

> 试图将一个只支持macOS、Linux、Windows平台的cpp项目迁移到Android平台，费了好大一番力气。

## 网上查到的方法

1. 使用Android Studio进行交叉编译
2. 使用cmake配置好```Android SDK```的```TOOLCHAIN```进行编译

这两种方法都尝试了，均遇到一点点小问题，而且都太杀鸡用牛刀了

## 简单方法

> 如果有像```g++ main.cpp```一样方便的编译方法该有多好

经Shino提醒：![shino](./shino.png) 为什么我要坚持和集成环境在这互相折磨

### NDK里的clang

华华这里使用的是macOS，使用win和linux环境要替换相应平台的命令行指令，如```$NDK_HOME -> %NDK_HOME%```

由于主播用的Android Studio，已经帮我下载好了NDK，路径：

* ```Android SDK: /Users/hh/Library/Android/sdk```
* ```Android NDK: /Users/hh/Library/Android/sdk/ndk/27.0.12077973```

这里设```ANDROID_NDK_HOME=/Users/hh/Library/Android/sdk/ndk/27.0.12077973```

然后，在```$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin```里，有各种llvm工具，包括```clang, ld``` 等等

```bash
(base) ➜  darwin-x86_64 ls bin | grep 35
aarch64-linux-android35-clang
aarch64-linux-android35-clang++
armv7a-linux-androideabi35-clang
armv7a-linux-androideabi35-clang++
i686-linux-android35-clang
i686-linux-android35-clang++
riscv64-linux-android35-clang
riscv64-linux-android35-clang++
x86_64-linux-android35-clang
x86_64-linux-android35-clang++
```

使用这个clang就可以编出Android的原生二进制了

### 编译

将clang的路径和ndk的路径添加到path

```bash
export PATH=$PATH:$ANDROID_NDK_HOME
export PATH=$PATH:$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin
```

然后```cd```回项目文件
```bash
aarch64-linux-android35-clang++ main.cpp -o hello
```
就编译出名为```hello```的可执行文件了

### 运行

使用adb，先有线连接手机

```bash
adb push hello /data/local/tmp
adb shell
# --- 进入 adb 环境 ---
cd /data/local/tmp
chmod +x hello
./hello
# --- 程序内容 ---
Hello, World!
```

## 关于前两种方法的反思

### Android Studio

我的目标是要一个可以直接执行的程序，不通过JNI接口调用，使用Android Studio只会编译出lib而不是exe

### cmake

虽然设置了```-DCMAKE_TOOLCHAIN_FILE```，但在外部使用```cmake```命令启动时，就已经是本地的cmake了，应该改成Android SDK里面附带的cmake，编译出的就是Android原生程序