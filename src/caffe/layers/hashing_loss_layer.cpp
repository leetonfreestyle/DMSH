#include <algorithm>
#include <vector>

#include "caffe/layers/hashing_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HashingLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  // diff_.Reshape(1, bottom[0]->channels(), 1, 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->num(), 2, 1);
  loss_.Reshape(bottom[0]->channels(), bottom[0]->num(), 1, 1);
  counter_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void HashingLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // initialize parameters
  const int num = bottom[0]->num();
  const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(num * (num - 1));
  const Dtype beta = top[0]->cpu_diff()[0] / static_cast<Dtype>(num);
  const int channels = bottom[0]->channels();
  const Dtype margin = this->layer_param_.hashing_loss_param().bi_margin();
  const Dtype tradeoff = this->layer_param_.hashing_loss_param().tradeoff();
  const int nsim = this->layer_param_.hashing_loss_param().nsim();
  const int label_num = bottom[1]->count() / num;
  bool sim;
  Dtype loss(0.0);
  Dtype reg(0.0);
  Dtype data(0.0);
  Dtype dist_sq(0.0);

  // adaptive sampling
  Dtype sim_ratio(0.0);
  Dtype count_total(0.0);
  Dtype count_sim(0.0);
  Dtype count_iter(0.0);
  Blob<Dtype> temp_;
  temp_.Reshape(num,num,2,1);
  Dtype* temp = temp_.mutable_cpu_data();
  caffe_rng_uniform<Dtype>(num * num * 2,Dtype(0.0),Dtype(1.0),temp);
  for (int i = 0; i < num; ++i) {
    for (int j=i+1; j < num; ++j) {
      temp[i * num * 2 + j * 2]= caffe_cpu_dot(label_num, 
        bottom[1]->cpu_data() + (i * label_num), bottom[1]->cpu_data() + (j * label_num));
      if (temp_.cpu_data()[i * num * 2 + j * 2] >= nsim)
        count_sim++;
      count_total++;
    }
  }
  sim_ratio = count_sim / count_total;
  Dtype limit_low(0.1);
  Dtype p_dis(1.1);
  if (sim_ratio < limit_low)
    p_dis = (sim_ratio / limit_low - sim_ratio) / (1 - sim_ratio);

  // calculate loss and gradient
  Dtype* bout = bottom[0]->mutable_cpu_diff();
  caffe_set(channels*num, Dtype(0), bout);
  for (int i = 0; i < num; ++i) {
    for (int j = i + 1; j < num; ++j){
      // sim = temp_.cpu_data()[i * num * 2 + j * 2] >= nsim;
      sim = temp_.cpu_data()[i * num * 2 + j * 2] >= Dtype(1.0);
      if (!sim && (temp_.cpu_data()[i * num * 2 + j * 2 + 1] > p_dis))
        continue;
      count_iter++;
      caffe_sub(
        channels,
        bottom[0]->cpu_data()+(i*channels),  // a
        bottom[0]->cpu_data()+(j*channels),  // b
        diff_.mutable_cpu_data());  // a_i-b_i
      dist_sq = caffe_cpu_dot(channels, diff_.cpu_data(), diff_.cpu_data());  //D_w^2
      if (sim) {  // similar pairs
        // loss += dist_sq;
        //Dtype sim_margin = channels * Dtype(4.0) * (
        //    std::pow(margin, - temp_.cpu_data()[i * num * 2 + j * 2]));
        // Dtype sim_margin = Dtype(4.0) * margin;
        sim_margin = channels * Dtype(4.0) * margin * (
            std::exp(- temp_.cpu_data()[i * num * 2 + j * 2]));
        //Dtype sim_margin = channels * Dtype(4.0) * (Dtype(1.0) / margin - Dtype(0.1) * temp_.cpu_data()[i * num * 2 + j * 2]);
        loss += std::max(dist_sq - sim_margin, Dtype(0.0));
        if ((dist_sq - sim_margin) >= Dtype(0.0)) {
          // gradient with respect to the first sample
          caffe_cpu_axpby(
            channels,
            alpha,
            diff_.cpu_data(),
            Dtype(1.0),
            bout + (i*channels));
          // gradient with respect to the second sample
          caffe_cpu_axpby(
            channels,
            -alpha,
            diff_.cpu_data(),
            Dtype(1.0),
            bout + (j*channels));
        }
      } 
      else {  // dissimilar pairs
        loss += std::max(channels * Dtype(2.0) - dist_sq, Dtype(0.0));
        if ((channels * Dtype(2.0) - dist_sq) > Dtype(0.0)) {
          // gradient with respect to the first sample
          caffe_cpu_axpby(
            channels,
            -alpha,
            diff_.cpu_data(),
            Dtype(1.0),
            bout + (i*channels));
          // gradient with respect to the second sample
          caffe_cpu_axpby(
            channels,
            alpha,
            diff_.cpu_data(),
            Dtype(1.0),
            bout + (j*channels));
        }
      }
    }
    for (int k = 0; k < channels;k++){
      data = *(bottom[0]->cpu_data()+(i*channels)+k);
      // gradient corresponding to the regularizer
      *(bout + (i*channels) + k) += beta * tradeoff * (((data>=Dtype(1.0))||
        (data<=Dtype(0.0)&&data>=Dtype(-1.0)))?Dtype(1.0):Dtype(-1.0));
      data = std::abs(data)-1;
      reg += std::abs(data);
    }
  }
  loss = loss / count_iter;
  loss += tradeoff * reg /static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HashingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(HashingLossLayer);
#endif

INSTANTIATE_CLASS(HashingLossLayer);
REGISTER_LAYER_CLASS(HashingLoss);

}  // namespace caffe
