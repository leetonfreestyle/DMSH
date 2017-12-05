import caffe
import numpy as np

class WmapLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.N      = bottom[0].shape[0]
        self.nbit   = bottom[0].shape[1]
        self.nlabel = bottom[1].shape[1]
        self.B      = np.zeros((self.N, self.nbit),np.float64)
        self.label  = np.zeros((self.N, self.nlabel),np.int64)
        self.testlabel = np.arange(self.N, dtype = np.int64).reshape(self.N, 1)

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def count_same_label(self,x,y):
        return np.dot(self.label[x],self.label[y])

    def forward(self, bottom, top):
        self.B      = np.sign(bottom[0].data[...])
        self.label  = bottom[1].data[...].reshape(self.N, self.nlabel)
        dis_mtx     = - np.dot(self.B,self.B.T)
        query_label = np.array(self.testlabel)
        mAP         = np.zeros((self.N, 1))
        label_mtx   = np.tile(self.testlabel,(1,self.N))
        idx_mtx     = np.argsort(dis_mtx, 0)
        sorted_label_mtx = np.zeros((self.N,self.N),np.int64)
        
        for q in xrange(self.N):
            sorted_label_mtx[:,q] = label_mtx[np.ix_(idx_mtx[:, q], [q])].reshape(self.N)
        # element-wise function
        vectorizefunc = np.vectorize(self.count_same_label)
        slabel_mtx = vectorizefunc(sorted_label_mtx,np.tile(self.testlabel.T,(self.N,1)))
        # Weighted mAP calculation
        slabel_sum_mtx = np.dot(np.tril(np.ones((self.N,self.N))),slabel_mtx)
        for q in xrange(self.N):
            Qi = (slabel_mtx[:, q] >= 1).sum()
            idx_relevant = np.where(slabel_mtx[:, q] >= 1)[0]
            mAP[q] = np.sum(slabel_sum_mtx[np.ix_(idx_relevant, [q])].reshape(Qi) / 
                (idx_relevant + 1)) / Qi

        top[0].data[...] = np.mean(mAP)

    def backward(self, top, propagate_down, bottom):
        pass
