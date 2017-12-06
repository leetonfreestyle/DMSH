#encoding=utf-8

import numpy as np
import struct
import argparse
import sys
import gc

parser = argparse.ArgumentParser()
parser.add_argument("code")
parser.add_argument("label")
parser.add_argument("-nbit",type=int,default=12)
parser.add_argument("-nlabel",type=int,default=21)
parser.add_argument("-nbatch",type=int,default=10)
parser.add_argument("-batchsize",type=int,default=1000)
args = parser.parse_args()
print args

class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, nbit, nlabel, N):
        super(Evaluator, self).__init__()
        self.nbit   = nbit
        self.nlabel = nlabel
        self.N      = N
        self.B      = np.zeros((self.N, self.nbit),np.float64)
        self.label  = np.zeros((self.N, self.nlabel),np.int64)

    def parseFiles(self, codeFile, labelFile):
        with open(codeFile,"rb") as f_code,open(labelFile,"rb") as f_label:
            for i in xrange(self.N):
                for j in xrange(self.nbit):
                    self.B[i,j] = struct.unpack("f",f_code.read(4))[0]
                for j in xrange(self.nlabel):
                    self.label[i,j] = struct.unpack("f",f_label.read(4))[0]
        self.testlabel = np.arange(self.N, dtype = np.int64).reshape(self.N, 1)
        self.B = np.sign(self.B)

    def count_same_label(self,x,y):
        return np.dot(self.label[x],self.label[y])

    def compute_mAP_multi(self):
        print "distance caculation"
        dis_mtx = - np.dot(self.B,self.B.T)
        idx_mtx = np.argsort(dis_mtx, 0)
        del dis_mtx
        np.savetxt("binaryCode.txt", self.B, fmt="%f", delimiter=",")
        del self.B
        gc.collect()
        sorted_label_mtx = np.zeros((self.N,self.N),np.int64)
        for q in xrange(self.N):
            sorted_label_mtx[:,q] = self.testlabel[np.ix_(idx_mtx[:, q], [0])].reshape(self.N)
        np.savetxt("idx_mtx.txt", idx_mtx[:10,:], fmt="%d", delimiter=",")
        del idx_mtx
        gc.collect()
        print "element-wise function"
        for i in xrange(self.N):
            print i,'\r',
            sys.stdout.flush()
            for j in xrange(self.N):
                x = sorted_label_mtx[j,i]
                sorted_label_mtx[j,i] = self.count_same_label(x,i)
        slabel_mtx = sorted_label_mtx
        print "mAP calculation"
        # slabel_sum_mtx = np.zeros((self.N,self.N),np.int64)
        # for i in xrange(self.N):
        #     print i,'\r',
        #     sys.stdout.flush()
        #     total = 0
        #     for j in xrange(self.N):
        #         total += slabel_mtx[j,i]
        #         slabel_sum_mtx[j,i] = total
        slabel_sum_mtx = np.dot(np.tril(np.ones((self.N,self.N))),slabel_mtx)
        mAP = np.zeros((self.N, 1))
        for q in xrange(self.N):
            Qi = (slabel_mtx[:, q] >= 1).sum()
            idx_relevant = np.where(slabel_mtx[:, q] >= 1)[0]
            mAP[q] = np.sum(slabel_sum_mtx[np.ix_(idx_relevant, [q])].reshape(Qi) / 
                (idx_relevant + 1)) / Qi
        del slabel_sum_mtx
        gc.collect()
        print "Weighted mAP : %f" % np.mean(mAP)
        for nsim in xrange(1,4):
            counter_valid = 0
            result_mtx = slabel_mtx >= nsim
            for q in xrange(self.N):
                Qi = np.sum(result_mtx[:, q])
                if not Qi:
                    mAP[q] = 0.0
                    continue
                mAP[q] = np.sum(np.arange(1, Qi + 1, dtype=np.float64) / 
                    (np.where(result_mtx[:, q] == 1)[0] + 1)) / Qi
                counter_valid += 1
            print "mAP of %d labels : %f" % (nsim, np.sum(mAP) / counter_valid)

def main():
    e = Evaluator(args.nbit, args.nlabel, args.nbatch * args.batchsize)
    e.parseFiles(args.code,args.label)
    e.compute_mAP_multi()

if __name__ == '__main__':
    main()
