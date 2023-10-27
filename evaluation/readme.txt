本文件夹中代码和脚本用来生成dcm的图像anchor，代码情况如下：
1. codec_multiprocess.sh和codec_multiprocess.py
    这部分代码调用了VTM编解码器，进行编码和解码过程，VTM版本为VTM-13.0
    使用方法： bash codec_multiprocess.sh

2. calculate_bpp.py
    用来统计编码后码流文件的大小、原始图像像素数，以及根据这两部分计算得到码率大小
    使用方法：
        命令行命令： python calculate_bpp.py --image_path {} --bin_path {}
        其中image_path指的原始图像存放地址，bin_path指的码流文件存放地址
    输出：
        (1) 命令行窗口输出图像数目、像素数总和、码流文件大小总和、码率
        (2) ./output/bpp_info.csv 不仅保存了(1)中的全部信息，还保存了各幅图像的具体信息
3. inference.sh和inference.py
    利用detectron2推理解码得到的图像，detectron2的版本为detectron2-0.2.1
    使用方法：
    (1) 根据需要修改inference.sh脚本中的地址、数据集、任务等内容
    (2) 命令行命令： sh inference.sh
    输出：
        命令行窗口可以查看任务性能