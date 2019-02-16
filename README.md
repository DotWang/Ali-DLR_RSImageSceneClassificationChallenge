### 小记一下
#### R1
萌新一枚，快12月底经[@vicchu](https://github.com/vicchu/)介绍开始打比赛 ，因为之前大家都比较忙事儿很多，之前也顾不上做，正式开始搞已经很迟了，不过咱也是第一次参加刚入门啥都不懂，花了一个星期时间调学习率，瞎改模型，各种续命trn和val也上不去，val更是差的要死，
结束前才意识到数据归一化错了，强行挽回残局，最后把之前跑的一堆辣鸡模型集成起来，混进复赛，两个星期时间就这么糊里糊涂过去了，最终130/1327。
#### R2
稍微知道怎么回事了，也制定了计划（计划赶不上变化)，5-fold搞了一下做local validation，不过依然对S1理解不够，log+去噪+F-R分解，S2的话NDVI+NDWI, 
策略单通道+双通道+大小类，一律ResNet50（换了个专门搞32*32的，讲真早点拉我还能搞搞别的net），数据增强花样更多一点，TTA换成软voting，
本来愉快的炼丹tmd大年初四才意识到没搞去雾，亡羊补牢，临时补救一波，重trn了将近三分之二，单模0.79->0.82，最后集成一波，最终50/1327，
就这样吧（S1玩不转估计也上不去了）。
#### R3
坐看前排大佬秀操作，学习一波！


感谢[@vicchu](https://github.com/vicchu/)一直以来的并肩作战  
感谢[@YonghaoXu](https://github.com/YonghaoXu)提供的大力支持与指导

### WorkFlow

The challenge is identified as a Remote Sensing Image Scene Classification task. First, the Data pre-processing were adopted, The log transformation and Gaussian Filter are used on s1 SAR data to denoise, the Freeman-Durden decomposition is employed to generate the Pseudo-RGB bands so as to process more easily for networks, while the Dark Channel Prior method is used to defogging on s2 MSI data. The NDVI and NDWI are also computed to improve the competence of networks for recognizing the vegetation and water. Then, three strategies including single channel, double channel and Multistage-Classification which involve the ResNet 50 with Stratified 5-fold Cross Validation are operated to promote the diversity of the model. Also, TTA(test time Augmentation), soft-voting and other tricks are implemented on inference period.
