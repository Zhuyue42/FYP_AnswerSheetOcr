##将 oriImg_path中每 imgNum张图片合并保存到 mergeImg_path/[sortNum].jpg
# imgNum张图片组成随机数字序列，并生成数字序列字典文件 merge_numDic_[mode].pkl，供比对准确率及box矫正
#
#使用说明：
#   修改两个 path
#   修改 mode和 mImgNum 
#   如有需要可以修改图片大小及 imgNum
import cv2
import os
import numpy as np
import random
import pickle

def merge(oriImg_path, mergeImg_path
          , mode, mImgNum, imgNum=10,l=20, w=20):
    '''
    ----------------------------------------------------------------
    *参数说明：
    oriImg_path       源图像（待合成图像）路径
    mergeImg_path     合并后图像路径
    mode              合并模式，由于train目录和test不一样，所以选不同模式
    mImgNum           合并后图像数目，例如总图像3500，每张10，则有350，每个数字也各有350张，编号从0开始，最后一张编号为349
    imgNum            每张合并图图像数目，默认10
    l,w               每张源图像的长宽
    ----------------------------------------------------------------
    '''
    merge_img=np.zeros( (w,l*imgNum,3) , dtype=np.uint8 )       #新建合并空文件。np.zeros(shape,dtype数据类型)返回来一个给定形状和类型的用0填充的数组
    merge_numDic={}                                             #存放合并图像数字序列字典

    print('开始合并 '+mode+' 文件到'+mergeImg_path)
    if not os.path.exists(mergeImg_path):
        os.mkdir(mergeImg_path)
    #每 imgNum张图片合并为1张，共mImgNum张
    #每张图像数字序列随机，不按0123...这么规律（主要考虑训练时过去单一效果会不会差？）
    #数字被抽出后从labels删除，直到取完
    for ordNum in range(mImgNum):
        labels=[0,1,2,3,4,5,6,7,8,9]                            #图像标签列表
        merge_num = ''                                          #合并图像数字序列,例如2469710358
        for n in range( imgNum ):
            index_label = random.randint(0,imgNum-n-1)          #图像标签的索引[0,9]，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
            label = labels[index_label]                         #随机 图像标签
            merge_num += str(label)                             #拼接 数字序列

            img_path = oriImg_path + str(label) + ('/' if(mode=='train') else '_') + str(ordNum) + '.jpg'
            img = cv2.imread(img_path)
            merge_img[ 0:20, 20*n:20*(n+ 1) ] = img  # 图像添加到merge_img
            
            del labels[index_label]
        merge_numDic[ordNum] = merge_num
        cv2.imwrite( mergeImg_path + str(ordNum) + '.tif', merge_img)

    if not os.path.exists('D:/Year4/FYP/Project/train_data/digit_B/data_merge/Dict'):
        os.mkdir('D:/Year4/FYP/Project/train_data/digit_B/data_merge/Dict')
    output = open('D:/Year4/FYP/Project/train_data/digit_B/data_merge/Dict/merge_numDic_'+mode+'.pkl', 'wb')    #创建导出文件
    pickle.dump(merge_numDic, output)                           #导出合并图像数字序列字典
    output.close()

    print(mode+'文件合并完成，已生成字典文件 D:/Year4/FYP/Project/train_data/digit_B/data_merge/Dict/merge_numDic_'+mode+'.pkl')
    #pkl_file = open('./merge_numDic_test.pkl', 'rb')               #使用pickle模块从文件中重构python对象
    #data1 = pickle.load(pkl_file);    print(data1)
    #pkl_file.close()



if __name__ == '__main__':
    
    #训练数据合并
    merge(oriImg_path = 'D:/Year4/FYP/Project/train_data/digit_B/data_train/'
          ,mergeImg_path = 'D:/Year4/FYP/Project/train_data/digit_B/data_merge/mergeData_train/'
          ,mode='train',mImgNum=350)
    #测试数据合并
    merge(oriImg_path = 'D:/Year4/FYP/Project/train_data/digit_B/data_test/'
          ,mergeImg_path = 'D:/Year4/FYP/Project/train_data/digit_B/data_merge/mergeData_test/'
          ,mode='test',mImgNum=150)
    input('已全部合并完成！按回车退出...')
