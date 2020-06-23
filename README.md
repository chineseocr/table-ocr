# 运用unet实现对文档表格的自动检测，表格重建 

## 实现功能   
- [x]  支持GPU，CPU（opencv dnn加速）；
- [ ]  整合darknet-ocr完成对表格的重建，输出json\excel
 
 
##  编译对GPU的支持   
`
git clone https://github.com/pjreddie/darknet.git ../darknet
cp Makefile ../darknet
cd ../darknet && make
`
 
## 下载text.weights模型文件   
模型文件地址:
http://59.110.234.163:9990/static/models/table-ocr/table.weights  
拷贝table.weights文件到models目录

## 测试
``` Bash
python3 table.py -jpgPath test/dd.jpg 
```

## 识别结果展示
<img width="500" height="300" src="https://github.com/chineseocr/table-ocr/blob/master/test/dd.jpg"/>  
### 横线竖线检测
<img width="500" height="300" src="https://github.com/chineseocr/table-ocr/blob/master/test/dd_seg.png"/>   
### 单元格输出
<img width="500" height="300" src="https://github.com/chineseocr/table-ocr/blob/master/test/dd_box.jpg"/>   
 
## 参考   
1. darket         https://github.com/pjreddie/darknet.git                 
2. darknet-ocr    https://github.com/chineseocr/darknet-ocr.git   
3. chineseocr     https://github.com/chineseocr/chineseocr.git       

## 技术支持  
mail:chineseocr@hotmail.com   
wechat:lywen52   
