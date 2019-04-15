1. 安装环境
2. config.py HOME_PATH_MAP中加入服务器环境信息 HOME_PATH_MAP={socket_name:home_path}
3. 生成HOME_PATH路径文件夹: 
    home_path/workspace/competitionn/population_prediction_JD
    并将该工程拷到home_path/workspace/competitionn/population_prediction_JD/目录下
4.  config.py 中 随意修改一下_C.MODEL.NAME名称
    sh train_glu_point.sh --gpu #生成 smgluresult_file, 在log/Experiment_name/result中
5.  config.py 中 随意修改一下_C.MODEL.NAME名称， 保证跟上一个模型名称不同
    sh train_biglu_point.sh --gpu #生成 bwgluresult_file,  在log/Experiment_name/result中
6.  cd utils 
    prediction_fusion.py中修改一下文件名：
    smgluresult = smgluresult_file的路径
    bwgluresult = bwgluresult_file的路径

    python3 prediction_fusion.py
    生成最终result,在当前目录的../../prediction_results/fusion/result.csv中