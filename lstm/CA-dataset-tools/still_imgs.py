from __future__ import division

import os
import cPickle
import numpy as np 
import scipy as scp
from scipy import ndimage
from tqdm import tqdm
import PIL
from PIL import Image

PATH='/mnt/data/datasets/collective-activity'

def unpickle(file_name):
    with open(file_name, 'rb') as fo:
        data = cPickle.load(fo)
    return data

def get_dir_list(path_to_dataset):
    files = os.listdir(path_to_dataset)
    dirs = []
    for entry in files:
        if os.path.isdir(path_to_dataset + '/' + entry):
            dirs.append(entry)

    dirs.sort()
    return dirs

def organize_training_data(path_to_dataset):
    #Get list of directories within dataset path.
    dirs = get_dir_list(path_to_dataset)
    print(dirs)
    #Iterate through each file in each directory, appending each file's path to a list
    path_list = []
    for entry in dirs:
        cur_path = path_to_dataset + '/' + entry
        files = os.listdir(cur_path)
        files.sort()
        for filename in files:
            if filename.endswith('.jpg'):
                path_list.append(cur_path + '/' + filename)

    #Save the list to a file.
    f = open(path_to_dataset + '/' + 'image_list.dat', 'wb')
    cPickle.dump(path_list, f)
    f.close()

def crop_still_imgs(path_to_dataset):
    output_path = path_to_dataset + '/stills/'
    dirs = get_dir_list(path_to_dataset+'/raw')
    # For each directory:
    imgs = []
    labels = []
    poses = []

    for d in dirs:
        cur_dir = path_to_dataset + '/raw/' + d
        with open(cur_dir+'/annotations.txt') as fo:
            print(d)
            for idx, line in tqdm(enumerate(fo)):
                strings = line.split()

                if len(strings[0]) == 3:
                    num = '0'+strings[0]
                else:
                    num = strings[0]
                
                image_path = cur_dir+'/frame'+num+'.jpg'
                tlc = (int(strings[1]), int(strings[2]))
                size = (int(strings[3]), int(strings[4]))
                label = int(strings[5])
                pose = int(strings[6])

                with open(image_path) as img:
                    array = scp.misc.imread(img)
                    if tlc[1] < 0:
                        tlc = (0, tlc[0])
                    if tlc[0] < 0:
                        tlc = (tlc[1], 0)

                    crop = array[tlc[1]:tlc[1]+size[1], tlc[0]:tlc[0]+size[0]]
                    if crop.shape[1] == 0:
                        print(d),
                        print(': %d: zero width: %s' %(idx, line))

                imgs.append(crop)
                labels.append(label)
                poses.append(pose)
    
    data = {'images': imgs,
            'labels': labels,
            'poses': poses}

    f = open(output_path + 'cropped', 'wb')
    cPickle.dump(data, f)
    f.close()

def analyze_dataset(path_to_dataset):
    data = unpickle(path_to_dataset)
    widths = {}
    heights = {}
    ratios = {}

    for idx, img in enumerate(data['images']):
        height = img.shape[0]
        width = img.shape[1]
        try:
            ratio = height / width
            ratio = int(ratio * 10)
        except ZeroDivisionError:
            print("Divide by zero in image %d. Shape: " %idx), 
            print(img.shape)
            ratio = 9999

        if height in heights:
            heights[height] += 1
        else:
            heights[height] = 1

        if width in widths:
            widths[width] += 1
        else:
            widths[width] = 1

        if ratio in ratios:
            ratios[ratio] += 1
        else:
            ratios[ratio] = 1

    heightkeys = []
    heightvalues = []
    for key, value in heights.items():
        heightkeys.append(key)
        heightvalues.append(value)
    heightkeys, heightvalues = zip(*sorted(zip(heightkeys, heightvalues)))
    for i in range(len(heightkeys)):
        print("%02d: %d" %(heightkeys[i], heightvalues[i]))

    widthkeys = []
    widthvalues = []
    for key, value in widths.items():
        widthkeys.append(key)
        widthvalues.append(value)
    widthkeys, widthvalues = zip(*sorted(zip(widthkeys, widthvalues)))
    for i in range(len(widthkeys)):
        print("%02d: %d" %(widthkeys[i], widthvalues[i]))

    ratiokeys = []
    ratiovalues = []
    for key, value in ratios.items():
        ratiokeys.append(key)
        ratiovalues.append(value)
    ratiokeys, ratiovalues = zip(*sorted(zip(ratiokeys, ratiovalues)))
    for i in range(len(ratiokeys)):
        print("%02d: %d" %(ratiokeys[i], ratiovalues[i]))

    h_idx = idx_max(heightvalues)
    w_idx = idx_max(widthvalues)

    max_h = heightkeys[h_idx]
    max_w = widthkeys[w_idx]

    print('mode of heights, widths: %d, %d' %(max_h, max_w))

def idx_max(values):
    return values.index(max(values))

def resize_cropped_imgs(path_to_dataset):
    target_height = 260
    target_width = 130
    data = unpickle(path_to_dataset+'cropped')
    img_res = []
    for img in tqdm(data['images']):
        pil_img = scp.misc.toimage(img)
        img_res.append(np.array(pil_img.resize((target_width, target_height), PIL.Image.ANTIALIAS)))

    new_data = {'images': img_res,
                'labels': data['labels'],
                'poses': data['poses']}

    f = open(path_to_dataset + 'resized', 'wb')
    cPickle.dump(new_data, f)
    f.close()

def main(unused_argv):
    #crop_still_imgs(PATH)
    resize_cropped_imgs(PATH+'/stills/')
    #analyze_dataset(PATH+'/stills/cropped')
    
if __name__ == '__main__':
    main(0)