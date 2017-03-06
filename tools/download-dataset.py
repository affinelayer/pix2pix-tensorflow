from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen # python 3
import tarfile
import tempfile
import shutil
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("dataset", default="facades", help="name of the dataset to download: cityscapes edges2handbags"
                    "edges2shoes, facades, maps")
parser.add_argument("--save_dir", default=".", help="path to save the images")
a = parser.parse_args()

url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % a.dataset
with tempfile.TemporaryFile() as tmp:
    print("downloading", url)
    shutil.copyfileobj(urlopen(url), tmp)
    print("extracting")
    tmp.seek(0)
    tar = tarfile.open(fileobj=tmp)
    os.chdir(a.save_dir)
    tar.extractall()
    tar.close()
    print("done")
