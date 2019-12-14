#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:51:10 2019
table
@author: chineseocr
"""
import argparse
def parser():
    parser = argparse.ArgumentParser(description="table lines detect")
    parser.add_argument('-jpgPath', help='image file')
    
    return parser.parse_args()

from image import letterbox_image,exp,minAreaLine,draw_lines,minAreaRectBox,draw_boxes,line_to_line,sqrt
from config import tableNetPath,GPU,SIZE
import numpy as np
import cv2
if not GPU:
     tableNet = cv2.dnn.readNetFromDarknet(tableNetPath.replace('.weights','.cfg'),tableNetPath)
else:
     from darknet import  load_net,predict_image,array_to_image
     
     tableNet = load_net(tableNetPath.replace('table.weights','table-darknet.cfg').encode(),tableNetPath.encode(), 0)
     
     
   
def dnn_table_predict(img,prob=0.5):   
    
    imgResize,fx,fy,dx,dy = letterbox_image(img,SIZE)
    
    imgResize = np.array(imgResize)
    imgW,imgH = SIZE
    image = cv2.dnn.blobFromImage(imgResize,1,size=(imgW,imgH),swapRB=False)
    image = np.array(image)/255
    tableNet.setInput(image)
    out=tableNet.forward()
    out = exp(out[0])
    out = out[:,dy:,dx:]
    return out,fx,fy


def darknet_GPU_predict(img,prob=0.5):
    imgW,imgH = SIZE
    imgResize,fx,fy,dx,dy = letterbox_image(img,SIZE)
    im  = array_to_image(imgResize)
    out = predict_image(tableNet, im)
    values = []
    for i in range(2*imgW*imgH):
                values.append(out[i])
    out = np.array(values).reshape((2,imgH,imgW))
    #out = exp(out)
    out = out[:,dy:,dx:]
    return out,fx,fy



    
    
    

from skimage import measure
def get_table_rowcols(img,prob,row=100,col=100):
    if not GPU:
        out,fx,fy = dnn_table_predict(img,prob)
    else:
        out,fx,fy = darknet_GPU_predict(img,prob)
        
    rows = out[0]
    cols = out[1]
    
    labels=measure.label(rows>prob,connectivity=2)
    regions = measure.regionprops(labels)
    RowsLines = [minAreaLine(line.coords) for line in regions if line.bbox[3]-line.bbox[1]>row ]
    
    labels=measure.label(cols>prob,connectivity=2)  
    regions = measure.regionprops(labels)
    ColsLines = [minAreaLine(line.coords) for line in regions if line.bbox[2]-line.bbox[0]>col ]

    
    tmp =np.zeros(SIZE[::-1],dtype='uint8')
    tmp = draw_lines(tmp,ColsLines+RowsLines,color=255, lineW=1)
    labels=measure.label(tmp>0,connectivity=2)  
    regions = measure.regionprops(labels)
    
    for region in regions:
        ymin,xmin,ymax,xmax = region.bbox
        label = region.label
        if ymax-ymin<20 or xmax-xmin<20:
            labels[labels==label]=0
    labels=measure.label(labels>0,connectivity=2)  
    
    indY,indX = np.where(labels>0)
    xmin,xmax = indX.min(),indX.max()
    ymin,ymax = indY.min(),indY.max()
    RowsLines = [p for p in RowsLines if xmin<=p[0]<=xmax and xmin<=p[2]<=xmax and ymin<=p[1]<=ymax and ymin<=p[3]<=ymax ]
    ColsLines = [p for p in ColsLines if xmin<=p[0]<=xmax and xmin<=p[2]<=xmax and ymin<=p[1]<=ymax and ymin<=p[3]<=ymax ]
    RowsLines = [[box[0]/fx,box[1]/fy,box[2]/fx,box[3]/fy] for box in  RowsLines]
    ColsLines = [[box[0]/fx,box[1]/fy,box[2]/fx,box[3]/fy] for box in  ColsLines]
    return RowsLines,ColsLines

def adjust_lines(RowsLines,ColsLines,alph=50):
    ##调整line 
    
    
    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines =[]
    newColsLines =[]
    for i in range(nrow):
        
        x1,y1,x2,y2 = RowsLines[i]
        cx1,cy1 = (x1+x2)/2,(y1+y2)/2
        for j in range(nrow):
            if i!=j:
                x3,y3,x4,y4 = RowsLines[j]
                cx2,cy2 = (x3+x4)/2,(y3+y4)/2
                if  (x3<cx1<x4 or y3<cy1<y4 ) or ( x1<cx2<x2 or y1<cy2<y2):
                    continue
                else:
                    r = sqrt((x1,y1),(x3,y3))
                    if r<alph:
                        newRowsLines.append([x1,y1,x3,y3])
                    r = sqrt((x1,y1),(x4,y4))
                    if r<alph:
                        newRowsLines.append([x1,y1,x4,y4])
                    
                    r = sqrt((x2,y2),(x3,y3))
                    if r<alph:
                        newRowsLines.append([x2,y2,x3,y3])
                    r = sqrt((x2,y2),(x4,y4))
                    if r<alph:
                        newRowsLines.append([x2,y2,x4,y4])   
                        
                        
    for i in range(ncol):
        x1,y1,x2,y2 = ColsLines[i]
        cx1,cy1 = (x1+x2)/2,(y1+y2)/2
        for j in range(ncol):
            if i!=j:
                x3,y3,x4,y4 = ColsLines[j]
                cx2,cy2 = (x3+x4)/2,(y3+y4)/2
                if  (x3<cx1<x4 or y3<cy1<y4 ) or ( x1<cx2<x2 or y1<cy2<y2):
                    continue
                else:
                    r = sqrt((x1,y1),(x3,y3))
                    if r<alph:
                        newColsLines.append([x1,y1,x3,y3])
                    r = sqrt((x1,y1),(x4,y4))
                    if r<alph:
                        newColsLines.append([x1,y1,x4,y4])
                    
                    r = sqrt((x2,y2),(x3,y3))
                    if r<alph:
                        newColsLines.append([x2,y2,x3,y3])
                    r = sqrt((x2,y2),(x4,y4))
                    if r<alph:
                        newColsLines.append([x2,y2,x4,y4]) 
                        
    return newRowsLines,newColsLines
                    
                 
                
                
                
                

def get_table_ceilboxes(img,prob,row=100,col=100,alph=50):
        """
        获取单元格
        """
        w,h =SIZE
        RowsLines,ColsLines=get_table_rowcols(img,prob,row,col)
        newRowsLines,newColsLines=adjust_lines(RowsLines,ColsLines,alph=alph)
        RowsLines = newRowsLines+RowsLines
        ColsLines = ColsLines+newColsLines

        nrow = len(RowsLines)
        ncol = len(ColsLines)
        
        for i in range(nrow):
            for j in range(ncol):
               
               RowsLines[i]=line_to_line(RowsLines[i],ColsLines[j],32)
               ColsLines[j]=line_to_line(ColsLines[j],RowsLines[i],32)
              
        
        tmp = np.zeros((img.size[1],img.size[0]),dtype='uint8')
        tmp = draw_lines(tmp,ColsLines+RowsLines,color=255, lineW=1)
        
        tabelLabels=measure.label(tmp==0,connectivity=2)  
        regions=measure.regionprops(tabelLabels)
        rboxes= []
        for region in regions:
            if region.bbox_area<h*w-10:
                rbox = minAreaRectBox(region.coords)
                rboxes.append(rbox)
            
        return rboxes,ColsLines,RowsLines


if __name__=='__main__':
    from PIL import Image
    import os
    config = parser()
    p= config.jpgPath
    if os.path.exists(p):
        img =Image.open(p).convert('RGB')
        rboxes,ColsLines,RowsLines = get_table_ceilboxes(img,prob=0.5,row=10,col=10,alph=10)
        tmp=draw_boxes(np.array(img),rboxes,(0,0,0))
        Image.fromarray(tmp).save(os.path.splitext(p)[0]+'_box.jpg')
        tmp = np.zeros((img.size[1],img.size[0]),dtype='uint8')
        tmp = draw_lines(tmp,ColsLines+RowsLines,color=255, lineW=1)
        cv2.imwrite(os.path.splitext(p)[0]+'_seg.png',tmp)
