import os
import json


import json
import os



def make_txtfile(filedir, savedir, imgsize):
    '''
    json파일을 읽어 라벨링 txt파일로 변환하는 함수
    filedir: json파일 경로
    savedir: txt파일 저장 경로
    '''
    filename=os.listdir(filedir)
    idx=1
    for file in filename:
        with open(filedir+file, 'r', encoding='utf-8') as f:
            a= json.load(f)
            with open(savedir+file[:-4]+'txt', 'w') as f:
                for i in a['annotations']:  
                    x, y, w, h = i['bbox']
                    # YOLO 포맷으로 변환 (normalized coordinates)
                    x_center = (x + w/2) / imgsize
                    y_center = (y + h/2) / imgsize
                    width = w / imgsize
                    height = h / imgsize
                    id= i['category_id']
                    f.write(f"{str(id)} {x_center} {y_center} {width} {height} \n")
        if idx%100 ==0:
            print(idx)
        idx+=1