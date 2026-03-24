---
title: "使用ncnn对RIFE模型进行int8量化加速"
subtitle: ""
date: 2025-04-23T18:04:17+08:00
toc:
    enable: true
weight: false
categories: ["人工智能"]
tags: ["ncnn", "RIFE", "量化加速"]
---

## 目的

笔者欲将rife模型在手机上完成部署，进行视频插帧推理。

项目：[https://github.com/zyhector/rife-ncnn-android](https://github.com/zyhector/rife-ncnn-android)

使用的rife-v4.6模型，优化了视频输入输出，解决磁盘IO的瓶颈后，在8Gen3手机上，720p下达到了5.5fps的推理速度。但还是挺慢，就试图借助量化来进行加速。

> 注意！！！
> 截止本篇成文之时（2025.04.23），ncnn不支持int8的Vulkan推理。int8量化后的模型只能使用CPU推理。[Issue Page](https://github.com/Tencent/ncnn/issues/5996#issuecomment-2812602978)

但做都做了，还是记录一下怎么做的int8量化。

## 事前准备

### ncnn

下载[ncnn](https://github.com/Tencent/ncnn)项目，并根据[ncnn wiki](https://github.com/Tencent/ncnn/wiki/how-to-build)编译对应平台的可执行程序。

### rife-ncnn-vulkan

下载[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)项目，备用。

### 数据集

笔者使用的[vimeo triplet](http://toflow.csail.mit.edu/)，训练RIFE使用的同款数据集。其他类似的图片数据集，或根据RIFE使用需求的特定图片数据集也可以。

## 量化流程

<center><img src="./flow chat.drawio.svg" width="75%" /></center>

主要分为两步

1. ```ncnn2table```，输入原模型、参考数据集，输出```.table```文件
2. ```ncnn2int8```，输入原模型、```.table```文件，输出量化后模型

> 后续操作文件夹为 ```ncnn/build/tools/quantize```，```ncnn2table```与```ncnn2int8```的所在文件夹。将数据集、模型文件都搬到这个文件夹里来。

## 数据集准备

根据```ncnn2table```的需求准备图片数据集。笔者使用的是vimeo_triplet中的一部分，将```vimeo_triplet/sequences/00001```整个文件夹拷贝到```quantize```文件夹下，```00001```中包含1000x3张图片，基本上够用了。

由于```rife```模型有三个输入，分别为图片输入in0，图片输入in1，timestep输入in2，因次我们要将输入分为三组，前两组便由数据集中的图片组成。

生成```imagelist.txt```：

```bash
find 00001 -type f | grep im1.png > im1list.txt
find 00001 -type f | grep im3.png > im3list.txt
```

## timestep准备

模型的in2输入接受的是插帧的时间点，这边就全部用0.5做输入，但由于ncnn2table只接受图片，得变通一下。

1. 更改ncnn2table代码，生成```[w,h,1]```的全为0.5的矩阵
2. 生成一张全为0.5的图片

这里用的第二种方法，生成一张```[1,1,1]```，值为0.5的jpg

```python
import cv2
import numpy as np

pixel_value = 0.5 * 255
image = np.array([[pixel_value]], dtype=np.uint8)

cv2.imwrite("timestep.jpg", image)
```

然后这个一像素的图片也放个[链接](./timestep.jpg)，有需者可以取之。

之后，手动生成一个```timestep.txt```，内含一千行的```timestep.jpg```

```python
with open("timestep.txt", "w") as file:
    for _ in range(1000):
        file.write("timestep.jpg\n")
```

## Warp算子支持

在rife模型中，原作者使用了自定义的Warp算子，非ncnn支持的算子之一。直接导入模型会不支持。但所幸nihui大佬的rife-ncnn-vulkan项目，手写了一个ncnn的Warp算子。我们要做的是在```ncnn2table```这程序里，注册Warp算子，让模型能正常的导入和运行。

### 引入Warp定义
从```rife-ncnn-vulkan/src/```中，拷贝```rife_ops.h, warp.cpp```，到```ncnn/tools/quantize```下。

### 生成对应Shader

在```rife-ncnn-vulkan```中，nihui使用了一点cmake魔法来将```warp.comp```文件变成vulkan的shader，其中操作大概如下：

1. ```src/generate_shader_comp_header.cmake```文件，处理```.comp```文件，变为```.comp.hex.h```头文件：

```cmake
# must define SHADER_COMP_HEADER SHADER_SRC

file(READ ${SHADER_SRC} comp_data)

# skip leading comment

string(FIND "${comp_data}" "#version" version_start)
string(SUBSTRING "${comp_data}" ${version_start} -1 comp_data)

# remove whitespace

string(REGEX REPLACE "\n +" "\n" comp_data "${comp_data}")

get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

# text to hex

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.text2hex.txt "${comp_data}")
file(READ ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.text2hex.txt comp_data_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," comp_data_hex ${comp_data_hex})
string(FIND "${comp_data_hex}" "," tail_comma REVERSE)
string(SUBSTRING "${comp_data_hex}" 0 ${tail_comma} comp_data_hex)

file(WRITE ${SHADER_COMP_HEADER} "static const char ${SHADER_SRC_NAME_WE}_comp_data[] = {${comp_data_hex}};\n")
```

2. ```src/CMakeLists.txt```文件中，定义宏，并处理```.comp```文件，将生成的```.comp.hex.h```文件放在```${CMAKE_CURRENT_BINARY_DIR}```里，并添加到```include_directories```
中：

```cmake
macro(rife_add_shader SHADER_SRC)
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.comp.hex.h)

    add_custom_command(
        OUTPUT ${SHADER_COMP_HEADER}
        COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC} -DSHADER_COMP_HEADER=${SHADER_COMP_HEADER} -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_comp_header.cmake"
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC}
        COMMENT "Preprocessing shader source ${SHADER_SRC_NAME_WE}.comp"
        VERBATIM
    )
    set_source_files_properties(${SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)

    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_COMP_HEADER})
endmacro()

rife_add_shader(warp.comp)
rife_add_shader(warp_pack4.comp)
rife_add_shader(warp_pack8.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
```

因此，我们需要复现一下这个cmake魔法。

1. 将```rife-ncnn-vulkan/src```下的```warp.comp, warp_pack4.comp, warp_pack8.comp```，复制到```ncnn/tools/quantize```文件夹中。
2. 将```rife-ncnn-vulkan/src```下的```generate_shader_comp_header.cmake```，复制到```ncnn/tools/quantize```文件夹中。
3. 更改```ncnn/tools/quantize/CMakeLists.txt```文件：

将上面最近的一个代码块，全部粘贴到该```CMakeLists.txt```文件的最前面。

再更改以下几行：

```cmake
    if(OpenCV_FOUND)
        add_executable(ncnn2table ncnn2table.cpp warp.cpp) # 修改

        add_dependencies(ncnn2table generate-spirv) # 新增

        target_include_directories(ncnn2table PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(ncnn2table PRIVATE ncnn ${OpenCV_LIBS})
    elseif(NCNN_SIMPLEOCV)
        add_executable(ncnn2table ncnn2table.cpp warp.cpp) # 修改

        add_dependencies(ncnn2table generate-spirv) # 新增

        target_compile_definitions(ncnn2table PUBLIC USE_NCNN_SIMPLEOCV)
        target_link_libraries(ncnn2table PRIVATE ncnn)
    else()
        add_executable(ncnn2table ncnn2table.cpp imreadwrite.cpp warp.cpp) # 修改

        add_dependencies(ncnn2table generate-spirv) # 新增

        target_compile_definitions(ncnn2table PUBLIC USE_LOCAL_IMREADWRITE)
        target_link_libraries(ncnn2table PRIVATE ncnn)
    endif()
```

如此以来，我们就复现了这个神秘的cmake魔法，可以生成shader文件，并通过hex数据导入到Warp算子中。

**随后重新编译ncnn，生成修改后的ncnn2table程序！**

## 生成table文件

当前文件夹结构应当如下，请检查对应的文件是否都到位了：
```text
ncnn/build/tools/quantize
├── 00001
│   ├── 0001
│   ├── 0002
│   ├── ....
│   ├── 0999
│   └── 1000
├── rife-v4.6
│   ├── flownet.bin
│   └── flownet.param
├── ncnn2int8
├── ncnn2table
├── im1list.txt
├── im3list.txt
├── timestep.jpg
└── timestep.txt
```

然后我们使用```ncnn2table```：

```bash
./ncnn2table                                     \
    rife-v4.6/flownet.param                      \
    rife-v4.6/flownet.bin                        \
    im1list.txt,im3list.txt,timestep.txt         \
    rife-v4.6/flownet.table                      \
    mean=[0,0,0],[0,0,0],[0,0,0]                 \
    norm=[1,1,1],[1,1,1],[1,1,1]                 \
    shape=[1280,736,3],[1280,736,3],[1280,736,1] \
    pixels=RGB,RGB,GRAY                          \
    thread=1                                     \
    method=kl
```

有几个需要注意的点：

1. ```mean,norm```分别是传入给```ncnn2table```对数据进行预处理的值，即将```(原始数据 - mean) / norm```作为**Input**，输入到模型。我们这里不需要预处理，就指定```mean=0, norm=1```
2. ```shape```，注意到我们虽然使用的是```1280x720```的图片，但传进去的```shape```是```1280x736```，这是因为RIFE只接受**32n**的图片输入，才能正确的卷积。
3. ```shape```，timestep的shape也为```1280x736```，这是因为在正常的rife推理中，也是将timestep处理成和图像一样大小的矩阵，传给模型。在```ncnn2table```内部，会将我们```1x1```的```timestep.jpg```，resize成```1280x736```的矩阵，进行处理。
4. 笔者用的```zsh```，不支持```mean=[0,0,0]```这种参数输入，需要改用```bash```

运行结束后，在```rife-v4.6/```下，生成了```flownet.table```文件。

## 进行量化

到这一步反而没什么困难了：

```bash
./ncnn2int8                      \
    rife-v4.6/flownet.param      \
    rife-v4.6/flownet.bin        \ 
    rife-v4.6-int8/flownet.param \
    rife-v4.6-int8/flownet.bin   \
    rife-v4.6/flownet.table
```

量化完的模型位于```rife-v4.6-int8/```

## 运行量化模型

由于int8量化只能使用CPU推理，还要对代码进行一些修改才能正常运行。

1. CPU运行时，不进行```set_vuklan_device```
```cpp
// rife.cpp
    if (vkdev) {
        flownet.set_vulkan_device(vkdev);
        contextnet.set_vulkan_device(vkdev);
        fusionnet.set_vulkan_device(vkdev);
    }
```

2. 开启int8推理的option
```cpp
// rife.cpp
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = vkdev ? true : false;
    opt.use_fp16_storage = vkdev ? true : false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;
    opt.use_int8_inference = true; // 新增这一行
```

3. 开启ncnn中int8推理相关的组件，否则会缺少响应的```layer_creator```，导致段错误
```cmake
# 顶层CMakeLists.txt

set(NCNN_INT8 ON)
set(WITH_LAYER_quantize ON)
set(WITH_LAYER_requantize ON)
set(WITH_LAYER_dequantize ON)
set(WITH_LAYER_gemm ON)
```

重新编译rife-ncnn-vulkan，添加```-g -1```参数，成功运行量化后模型。

## 运行效率与精度损失

还没测......

会回来补的！