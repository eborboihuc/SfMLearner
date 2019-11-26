# coding: utf-8

from __future__ import division
import os
import argparse
import numpy as np
import PIL.Image as pil
import tensorflow as tf
import matplotlib.pyplot as plt

from SfMLearner import SfMLearner
from utils import normalize_depth_for_display


class Demo(object):
    '''
    Demo Wrapper
    '''

    def __init__(self,
            sess,
            file_path = 'misc/',
            subname = 'png',
            img_height = 128,
            img_width = 416,
            channels = 3,
            ckpt_file = 'models/model-190532'
            ):

        print("Initializing Demo...")
        self.sess = sess
        self.file_path = file_path
        self.subname = subname
        self.img_height = img_height
        self.img_width = img_width
        self.ckpt_file = ckpt_file
        self.channels = channels

        self.sfm = SfMLearner()
        self.sfm.setup_inference(
                self.img_height,
                self.img_width,
                mode='depth')
        
        saver = tf.train.Saver([var for var in tf.model_variables()]) 
        saver.restore(sess, self.ckpt_file)

    def forward(self, img_name):
        image = self.get_image(img_name)
        print(image.shape)
        pred = self.get_depth(image)
        self.save_image(
                img_name.replace('.{}'.format(self.subname), '_dep.{}'.format(self.subname), -1), 
                pred
                )

    def get_lists(self):
        print("Get Image List...")
        filelist = []
        for dirpath, _, _ in os.walk(self.file_path):
            filelist += [os.path.join(dirpath, item) for item in os.listdir(dirpath) if item.endswith(self.subname)]
        
        print(filelist)
        assert len(filelist) > 0, "Found 0 Images"
        
        return filelist

    def get_image(self, img_name):
        print("Open Image {}".format(img_name))
        img = pil.open(img_name)
        img = img.resize([self.img_width, self.img_height], pil.ANTIALIAS)
        img = np.array(img).astype('float32')
        img = np.expand_dims(img, axis = 0)[:, :, :, :self.channels]

        return img
 
    def get_depth(self, image):
        print("Convert Image...")
        return self.sfm.inference(image, self.sess, mode='depth')

    def save_image(self, img_name, image):
        print("Save File : {}".format(img_name))
        plt.imsave(img_name, normalize_depth_for_display(image['depth'][0,:,:,0]))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_path', help='Converted parameters for the model', default='models/model-190532')
    parser.add_argument('-f', dest='file_path', help='Directory of images to predict', default='../360_input')
    args = parser.parse_args()


    with tf.Session() as sess:
        SfM_demo = Demo(sess, file_path=args.file_path, ckpt_file=args.model_path, subname='png', img_width=480, img_height=480)

        filelist = SfM_demo.get_lists()

        for img_name in filelist:
            SfM_demo.forward(img_name)



if __name__ == '__main__':
    main()   

