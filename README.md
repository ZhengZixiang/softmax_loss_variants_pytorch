# loss_for_text_classification_pytorch
The PyTorch implementation of variants of loss for text classification & text matching.

基于PyTorch实现的文本分类与文本匹配损失函数变种

## Related Papers
- Distilling the Knowledge in a Neural Network (NIPS 2014 DeepLearning Workshop) [[paper]](https://arxiv.org/abs/1503.02531) - ***Soft Target & Soft Softmax Loss***
- FaceNet: A Unified Embedding for Face Recognition and Clustering (CVPR 2015) [[paper]](https://arxiv.org/abs/1503.03832) - ***Triplet Loss***
- Applying Deep Learning to Answer Selection: A Study and An Open Task (ASRU 2015) [[paper]](https://arxiv.org/abs/1508.01585)
- Holistically-Nested Edge Detection (ICCV 2015) [[paper]](https://arxiv.org/abs/1504.06375) - ***Weigted Softmax Loss***
- V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation (3DV 2016) [[paper]](https://arxiv.org/abs/1606.04797) - ***Dice Loss***
- UnitBox: An Advanced Object Detection Network (ACM Multimedia 2016) [[paper]](https://arxiv.org/abs/1608.01471) - ***IoU Loss***
- Rethinking the Inception Architecture for Computer Vision (CVPR 2016) [[paper]](https://arxiv.org/abs/1512.00567) - ***Label Smoothing***
- A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 2016) [[paper]](https://link.springer.com/chapter/10.1007%2F978-3-319-46478-7_31) - ***Center Loss***
- Large-Margin Softmax Loss for Convolutional Neural Networks (ICML 2016) [[paper]](https://arxiv.org/abs/1612.02295) - ***L-softmax Loss***
- SphereFace: Deep Hypersphere Embedding for Face Recognition (CVPR 2017) [[paper]](https://arxiv.org/abs/1704.08063) - ***A-softmax Loss***
- Focal Loss for Dense Object Detection (ICCV 2017) [[paper]](https://arxiv.org/abs/1708.02002) - ***Focal Loss***
- The Lovász-Softmax Loss: A Tractable Surrogate for The Optimization of The Intersection-over-Union Measure in Neural Networks (CVPR 2018) [[paper]](https://arxiv.org/abs/1705.08790) [[code]](https://github.com/bermanmaxim/LovaszSoftmax) - ***Lovasz Softmax Loss***
- Island Loss for Learning Discriminative Features in Facial Expression Recognition (FG 2018) [[paper]](https://arxiv.org/abs/1710.03144) - ***Island Loss***
- Feature Incay for Representation Regularization (ICLR 2018 Workshop) [[paper]](https://arxiv.org/abs/1705.10284)
- Additive Margin Softmax for Face Verification  (ICLR 2018 Workshop) [[paper]](https://arxiv.org/abs/1801.05599) [[code]](https://github.com/happynear/AMSoftmax) - ***AM-softmax Loss***
- AnatomyNet: Deep Learning for Fast and Fully Automated Whole-volume Segmentation of Head and Neck Anatomy (Medical Physics 2018) [[paper]](https://arxiv.org/abs/1808.05238) - ***Exponential Logarithmic Loss (Focal Loss + Dice Loss)***
- ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019) [[paper]](https://arxiv.org/abs/1801.07698) - ***AA-softmax Loss***
- Mixtape: Breaking the Softmax Bottleneck Efficiently (NeurIPS 2019) [[paper]](https://papers.nips.cc/paper/9723-mixtape-breaking-the-softmax-bottleneck-efficiently) - ***Mixtape***
- When Does Label Smoothing Help? (NeuIPS 2019) [[paper]](https://arxiv.org/abs/1906.02629)
- Complement Objective Training (ICLR 2019) [[paper]](https://arxiv.org/abs/1903.01182) - ***COT***
- Dice Loss for Data-imbalanced NLP Tasks (CoRR 2019) [[paper]](https://arxiv.org/abs/1911.02855) - ***Dice Loss***
- Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks (MLMI@MICCAI 2017) [[paper]](https://arxiv.org/abs/1706.05721) - ***Tversky Loss***

## Related Chinese Posts
- [label smoothing 如何让正确类与错误类在 logit 维度拉远的？](https://zhuanlan.zhihu.com/p/73054583)
- [如何理解soft target这一做法？ - 知乎](https://www.zhihu.com/question/50519680?sort=created)
- [有哪些「魔改」loss函数，曾经拯救了你的深度学习模型？ - 知乎](https://www.zhihu.com/question/294635686/answer/606259229)
- [图像分割中的loss--处理数据极度不均衡的状况](https://www.cnblogs.com/hotsnow/p/10954624.html)
- [【损失函数合集】超详细的语义分割中的Loss大盘点 - 言有三](https://www.jianshu.com/p/6328bf066061)
- [【技术综述】一文道尽softmax loss及其变种 - 言有三](https://zhuanlan.zhihu.com/p/34044634)
- [Softmax理解之从最优化的角度看待Softmax损失函数 - 王峰](https://zhuanlan.zhihu.com/p/45014864)
- [Softmax理解之二分类与多分类 - 王峰](https://zhuanlan.zhihu.com/p/45368976)
- [Softmax理解之Smooth程度控制 - 王峰](https://zhuanlan.zhihu.com/p/49939159)
- [Softmax理解之margin - 王峰](https://zhuanlan.zhihu.com/p/52108088)
- [Softmax理解之margin的自动化设置 - 王峰](https://zhuanlan.zhihu.com/p/62229855)
- [Softmax理解之被忽略的Focal Loss变种 - 王峰](https://zhuanlan.zhihu.com/p/62314673)
- [文本情感分类（四）：更好的损失函数 - 科学空间](https://kexue.fm/archives/4293)
- [从loss的硬截断、软化到focal loss - 科学空间](https://kexue.fm/archives/4733)
- [Keras中自定义复杂的loss函数 - 科学空间](https://kexue.fm/archives/4493)
- [基于GRU和am-softmax的句子相似度模型 - 科学空间](https://kexue.fm/archives/5743)
