                         本项目是基于文本无关的说话人确认
本说明书将从原理，数据处理，模型构建和代码测试使用来说明

一.原理
本项目的原理是基于LSTM的d-vector特征辨别，同时使用Generalized end-to-end loss来优化
主要参考论文如下：
2018 ICASSP--Google Inc-- Generalized End-to-End Loss for Speaker Verification 
详细图片见./images/原理1.png

首先使用lstm循环神经来提取多个不同说话人的多条音频序列数据的特征d-vector

其次直接使用文章提出的端到端的算法直接对这多个d-vector特征向量进行训练，使得来自同一个人的d-vector之间的距离
尽可能的小，而使来自不同的说话人的d-vector特征向量之间的距离尽可能的远离

这种方法不同于之前的d-vector方法，之前的方法通过前期进行分类的训练，取最后一个隐层向量作为d-vector，这是一种
间接的d-vector训练方式，本文是直接利用d-vector来进行训练，直接以说话人区分为目的，不在经过分类的这一过程。

二.数据处理
在该模型中主要是基于中文的说话人确认
该训练数据结合了多个数据集的音频, 具体组成如下：

dataset     Speakers    wav_files   Female/male
thchs30	    60	        40 hours 	f
aishell1	400	        178 hours	f and m
st-cmds	    855	        100 hours	f and m
aidatatang	600	        200 hours	f and m

对于每一个说话人的音频数据，我们这里将他们处理到一个大的npy文件里去，其size为：[N，L，D]
其中N表示这个说话人说了多少句话，L表示每句话处理之后的序列长度，D表示每一帧的向量表示维度
run: python data_preprocess.py

三.模型构建
本模型较为简单主要是三层的lstm层，一层的全连接fc层，其中，lstm的最后一个hidden输出作为整个序列的向量表示，在接两层全连接层
全连接层的输出为d-vector表示

四.模型训练
本项目模型训练的目标函数见./images/loss1.png 和loss2.png 
run: python main.py

五.模型测试

性能测试代码是test_eer.py, 这里主要是用EER来评价模型性能， 调节thres，使得FAR=FRR， 这时，EER=FAR=FRR
该模型测试注册数据为4时(即4条语音数据)， EER=14.3%
当注册数据增加时，20-30条时，EER=8.7%
run: python test_eer.py

精度测试代码是test_acc.py,这里主要是在实际中使用时的模型正确接受的精度，如，注册数据为一条音频时，该条音频通过模型得到一个向量表示d-enroll
当其他测试数据音频输入到模型中时，得到d-test, 则计算d-enroll和d-test的相似度
同时设置一个阈值thres, 当相似度大于该阈值时，模型判断接受，Accept, 相反，当相似度小于该阈值时，模型判断拒绝，Reject.
acc = n_accept / n_total
run: python test_acc.py

注册数据为1时，调节thres，acc精度如下：
threshold    acc
0.6          0.93
0.7          0.82
0.8          0.30
0.9          0.15

thres为0.85时，调节注册数据，acc精度如下：
enroll_utterances  acc
1                  0.20    
3                  0.48
5                  0.53
7                  0.64
10                 0.75
20                 0.78

thres = 0.75, entroll_utterances = 6, acc = 0.84
thres = 0.65, entroll_utterances = 6, acc = 0.91