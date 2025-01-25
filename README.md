# FedDG

<<<<<<< HEAD
```bash
python fed_dg.py --dataset PACS --test_domain photo --device 3
=======
#目录结构描述<br>
domain_clf.py           // 域分类器的实现
    
fed_dg.py    // 训练客户端和服务器端,并进行测试
    
utils.py            // 固定权重和动态权重聚合的实现 根据命令行数据集设定数据路径和超参数<br>

PS:我加入域分类器后,test acc离奇的低,还请学长麻烦看一下哪里除了问题;不加域分类器反而比较接近baseline<br>

#运行方式<br>
训练和测试使用同一个.py文件,如下所示
```bash
python fed_dg.py --dataset PACS --test_domain photo --device 0
>>>>>>> ad90ebb120f7499c014045848882beba26516ffb
