data_path = './data/202277'

# 一般不需要改的参数
scale_L23 = 0.82 #双光子成像分辨率
scale_L5 = 0.83
# scale = 0.83

# 识别的参数
conf_theta = 0.1 # 细胞识别的置信度
distance_thread = 8 # 两个细胞最近距离
batch_size = 2 # 根据显卡环境可以改
model_path = './deep_detection/model.pth'


# 匹配的参数
thickness_L23 = 30 # 双光子成像厚度
thickness_L5 = 70
theta = 20 #边相似度参数
step0 = 10 # 每次步进距离
# thickness1 = 50 #结果图中的投影厚度
