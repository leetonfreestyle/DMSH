import numpy as np
import sys
CAFFE_ROOT = "/home/leeton/dsh-master/"
sys.path.insert(1,CAFFE_ROOT + "python")
import caffe
import os

class CaffeTestNet(object):
    """docstring for CaffeTestNet"""
    def __init__(self, caffeRoot, deployPath, weightsPath, gpu):
        super(CaffeTestNet, self).__init__()
        self.caffeRoot = caffeRoot
        self.deployPath = deployPath
        self.weightsPath = weightsPath
        self.gpu = gpu

        caffe.set_mode_gpu()
        caffe.set_device(self.gpu)
        self.net = caffe.Net(self.caffeRoot + self.deployPath,     # defines the structure of the model
            self.caffeRoot + self.weightsPath,                     # contains the trained weights
            caffe.TEST)                                     # use test mode (e.g., don't perform dropout)
        
        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
        
    def getFeature(self, picPath, layerName):
        image = caffe.io.load_image(picPath)
        transformed_image = self.transformer.preprocess('data', image)
        # copy the image data into the memory allocated for the net
        self.net.blobs['data'].data[...] = transformed_image 
        # perform classification
        output = self.net.forward()
        result = output[layerName]  # the output probability vector for the first image in the batch
        return ''.join(map(lambda x:"1" if x > 0 else "0",result[0]))

def main():
    imgPath = sys.argv[1]
    filenames = os.listdir(imgPath)
    with open("hashCode.txt","w+") as outfile:
        deployPath = "leeton/DSH_48bit_deploy.prototxt"
        weightsPath = "leeton/snapshots/DSH_48bit_iter_150000.caffemodel"
        ct = CaffeTestNet(CAFFE_ROOT, deployPath, weightsPath, 2)
        count = 0
        for filename in filenames:
            result = ct.getFeature(os.path.join(imgPath,filename),'ip1')
            print >>outfile,"%s %s" % (filename,result)
            count += 1
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()
        print "hashCode Done!"


if __name__ == '__main__':
    main()
