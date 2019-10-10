https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

文件夹中`jigsaw-toxic-comment-classification-challenge`为源数据；以`.ipynb`后缀的两个文件为调试用的notebook，分步的结果在notebook中试验；以`.py`后缀的两个文件顾名思义为Python文件，一个为训练模型代码。一个为测试模型代码；`models_params_file`文件夹保存了训练模型后得到的参数，以便于在测试时使用。`submission.csv` 文件为Kaggle网站要求的提交文件，里面数据为模型对测试用例的测试结果。

模型整体参考了TextCNN的构建思路。将卷积神经网络用于NLP中。
#### 首次训练已完成，正待检验以及改善。 -- 2019/10/10
#### 将生成预测结果文件导入Kaggle相应页面测试，得分0.92，不理想。 -- 2019/10/11
#### 利用MXNet混合编程模式对代码进行修改，将自建类TextCNN从继承nn.Block改为nn.HybridBlock，附带将nn.Sequential类修改为nn.HybridSequential类、nn.flatten函数修改为nn.Flatten类的前进步计算、nd.concat函数修改为mxnet.sym.Concat方法。以上修改将原本纯命令式编程改为混合式变成，提高了运算速度。如果在最后训练循环体中加入计数变量，可以观察到训练速度以肉眼可见的速度增长了。 -- 2019/10/11
#### 接下来会尝试修改各项超参进行模型改进，以提高分数。另外，也尝试思考其他模型。
