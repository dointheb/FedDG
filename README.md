# FedDG

#目录结构描述<br>
domain_clf.py           // 域分类器的实现
    
fed_dg.py    // 训练客户端和服务器端,并进行测试
    
utils.py            // 固定权重和动态权重聚合的实现 根据命令行数据集设定数据路径和超参数<br>


#运行方式<br>
训练和测试使用同一个.py文件,如下所示
```bash
python fed_dg.py --dataset PACS --test_domain photo --device 0
