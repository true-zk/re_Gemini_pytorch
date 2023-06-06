# re_Gemini_pytorch
re implementation of Gemini by pytorch

1. gendata.py数据集格式，从acfg生成需要的数据  
2. model.py主要的模型，最重要的是那个embed函数
3. raw_graphs.py包含了从二进制代码提取出来的acfg的原始类，大多数代码都是僵尸代码，这个是原始作者写的，不用管，只看那个raw_graphs的类就行
4. utils.py训练，验证模型的一些工具
