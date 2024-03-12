# Image to Latex

Team Collaboration :
<br/>Nguyễn Huy Hoàng
<br/>Nguyễn Tông Quân
<br/>Nguyễn Viết Vũ 
<br/>Nguyễn Xuân Trình
<br/>Phạm Văn Trường

## Introduction

This respository implement the Seq2Seq Image to Latex architecture from paper “Image to Latex.” of Genthial, Guillaume. (2017).

## Architecture

This structure is based on Seq2Seq architecture, it use one Convolutional Encoder and one RNN Decoder.

- Convolution (only)
- Convolution with Row Encoder (BiLSTM)
- Convolution with Batch Norm
- ResNet 18 with Row Encoder (BiLSTM)
- ResNet 18 (only)


<div>
    <image src="https://deforani.sirv.com/Images/Github/Image2Latex/image2latex.png" />
</div>

## Dataset
### im2latex170k
- https://www.kaggle.com/datasets/rvente/im2latex170k

## Example
- <a href="https://www.kaggle.com/code/tuannguyenvananh/image2latex-resnetbilstm-lstm">ResNet Row Encoder</a>
