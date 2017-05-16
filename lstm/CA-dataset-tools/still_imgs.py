import os
import cPickle

def organize_training_data(path_to_dataset):
    #Get list of directories within dataset path.
    files = os.listdir(path_to_dataset)
    dirs = []
    for entry in files:
        if os.path.isdir(path_to_dataset + '/' + entry):
            dirs.append(entry)

    dirs.sort()
    print(dirs)
    #Iterate through each file in each directory, assigning each path an integer key in a dict.
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

def main(unused_argv):
    organize_training_data('/mnt/data/datasets/collective-activity')

if __name__ == '__main__':
    main(0)