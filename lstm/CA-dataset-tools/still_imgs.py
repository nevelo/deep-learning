import os
import cPickle
import numpy as np 
import scipy as scp
from scipy import ndimage
from tqdm import tqdm

PATH='/mnt/data/datasets/collective-activity'

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
            for line in tqdm(fo):
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
                    crop = array[tlc[1]:tlc[1]+size[1], tlc[0]:tlc[0]+size[0]]
                imgs.append(crop)
                labels.append(label)
                poses.append(pose)
    
    data = {'images': imgs,
            'labels': labels,
            'poses': poses}

    f = open(output_path + 'cropped', 'wb')
    cPickle.dump(data, f)
    f.close()

def main(unused_argv):
    crop_still_imgs(PATH)
    
if __name__ == '__main__':
    main(0)