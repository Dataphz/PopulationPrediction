1. 京东2018人口预测比赛(赛区决赛第二)
       
2. 模型说明：
    1.输入数据准备:
        1. flow in, flow out , dwell同时输入
        2. 加入了trend, season, weight属性
        3. 清洗数据处理（节假日）
    2. 堆叠5层的GLU模型，输入30天预测下一天。迭代预测n天
    3. 再用13个堆叠5层的GLU模型, 基于之前学习模型对每个城市的残差进行建模
    4. 用BiGLU（堆叠2层）,双序列输入对中间5天时间序列进行建模
    5. 两个结果融合得到最终结果
    
3.安装环境:
    1. config.py HOME_PATH_MAP中加入服务器环境信息 HOME_PATH_MAP={socket_name:home_path}
    2. 生成HOME_PATH路径文件夹: 
        home_path/workspace/competitionn/population_prediction_JD
        并将该工程拷到home_path/workspace/competitionn/population_prediction_JD/目录下
    3.  config.py 中 随意修改一下_C.MODEL.NAME名称
        sh train_glu_point.sh --gpu #生成 smgluresult_file, 在log/Experiment_name/result中
    4.  config.py 中 随意修改一下_C.MODEL.NAME名称， 保证跟上一个模型名称不同
        sh train_biglu_point.sh --gpu #生成 bwgluresult_file,  在log/Experiment_name/result中
    5.  cd utils 
        prediction_fusion.py中修改一下文件名：
        smgluresult = smgluresult_file的路径
        bwgluresult = bwgluresult_file的路径
        python3 prediction_fusion.py
        生成最终result,在当前目录的../../prediction_results/fusion/result.csv
