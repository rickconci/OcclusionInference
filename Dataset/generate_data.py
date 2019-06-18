import sys
sys.path.append('../')
from os.path import abspath
import matplotlib.pyplot as plt
#%matplotlib inline
from utils import shlex_cmd
from scipy.io import savemat
from io_mod import save_image_set, read_image_set, name_files, save_images_as_mat
import generate_mod
import Clutter_mod

import pickle

DIGITS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

n_samples = 10
clutter_list = []
for i in range(n_samples):
    clutter_list += [generate_mod.sample_clutter(font_set=['arial-bold'], n_letters=3)]


clutter_list = name_files('test', clutter_list=clutter_list)
save_image_set(clutter_list, 'test/test.csv')

loaded_clutter_list = read_image_set('test/test.csv')
#print([cl.get_character_list() for cl in clutter_list])
#print([cl.get_character_list() for cl in loaded_clutter_list])

fname_list = []
for i, cl in enumerate(clutter_list):
    cl.render_occlusion(fname="test/orig_{}".format(i))
    cl.render_occlusion(fname="test/inverse_{}".format(i), inverse=True)
    fname_list.append("test/orig_{}".format(i))
    fname_list.append("test/inverse_{}".format(i))

#fname_list = [cl.fname for cl in clutter_list]
images_dict = save_images_as_mat(abspath('test/test.mat'), clutter_list, (60,60), fname_list=fname_list, overwrite_wdir=True)


pickle.dump(images_dict, open( "images_dict.p", "wb" ) )

