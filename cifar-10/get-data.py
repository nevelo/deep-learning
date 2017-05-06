import urllib2
import os
from subprocess import call

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
directory = 'dataset'

# class DirectoryError(Exception):
#     def __init__(self, value):
#         self.value=value
#     def __str__(self):
#         return repr(self.value)

def main():
    filename = url.split('/')[-1]
    print("Tool to download CIFAR-10 dataset.")

    if not os.path.exists(directory):
        print("Creating directory.")
        os.makedirs(directory)
    elif os.path.isfile(directory):
        raise Exception('ERROR: File \'dataset\' must not exist in target directory.')

    print("\nDownloading CIFAR-10 data into dataset/ directory.")

    os.chdir(directory)
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    filesize = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s | Bytes: %s" %(filename, filesize))
    downloaded = 0
    blocksize = 2**13
    while True:
        buffer = u.read(blocksize)
        if not buffer:
            break
        downloaded += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (downloaded, downloaded * 100. / filesize)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()
    print("\n\nDownload successful. Extracting files...")
    call(["tar", "-xzvf", "cifar-10-python.tar.gz"])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
