import numpy as np
import tensorflow as tf
import cv2


"""
YOLO v1

"""


class YOLO(object):

    def __init__(self, weights_file,verbose=True):
        self.verbose = verbose # 显示详细信息

        # 检测参数
        self.grid = 7
        self.box = 2
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
        self.class_num = len(self.classes)
        self.conf_threshold = 0.2
        self.iou_threshold = 0.5
        # shaoe [7, 7, 2]
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.grid)]*self.grid*self.box),
                                              [self.box, self.grid, self.grid]), [1, 2, 0])  # shape（7,7,2）
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2]) # shape (7,7,2)

        self.sess = tf.Session() # 创建会话
        self._build_net()  # 构建卷积神经网络
        self._load_weights(weights_file)  # 加载权重文件


    def _build_net(self):

        if self.verbose:
            print("Start to build the network……")

        self.images = tf.placeholder(tf.float32, [None,448,448,3])

        net = self._conv_layer(self.images, 7, 2, 64, 1) # 输入张量，卷积核尺寸，步长，数量，层id
        net = self._maxpool_layer(net, 2, 2, 1) # 输入张量，池化核尺寸，步长，层id

        net = self._conv_layer(net, 3, 1, 192, 2)
        net = self._maxpool_layer(net, 2, 2, 2)

        net = self._conv_layer(net, 1, 1, 128, 3)
        net = self._conv_layer(net, 3, 1, 256, 4)
        net = self._conv_layer(net, 1, 1, 256, 5)
        net = self._conv_layer(net, 3, 1, 512, 6)
        net = self._maxpool_layer(net, 2, 2, 6)

        net = self._conv_layer(net, 1, 1, 256, 7)
        net = self._conv_layer(net, 3, 1, 512, 8)
        net = self._conv_layer(net, 1, 1, 256, 9)
        net = self._conv_layer(net, 3, 1, 512, 10)
        net = self._conv_layer(net, 1, 1, 256, 11)
        net = self._conv_layer(net, 3, 1, 512, 12)
        net = self._conv_layer(net, 1, 1, 256, 13)
        net = self._conv_layer(net, 3, 1, 512, 14)
        net = self._conv_layer(net, 1, 1, 512, 15)
        net = self._conv_layer(net, 3, 1, 1024, 16)
        net = self._maxpool_layer(net, 2, 2, 16)

        net = self._conv_layer(net, 1, 1, 512, 17)
        net = self._conv_layer(net, 3, 1, 1024, 18)
        net = self._conv_layer(net, 1, 1, 512, 19)
        net = self._conv_layer(net, 3, 1, 1024, 20)
        net = self._conv_layer(net, 3, 1, 1024, 21)
        net = self._conv_layer(net, 3, 2, 1024, 22)
        net = self._conv_layer(net, 3, 1, 1024, 23)
        net = self._conv_layer(net, 3, 1, 1024, 24) # 输出[None,7,7,1024]

        net = self._flatten(net) # 将输入张量展开为一维
        net = self._fc_layer(net, 512, 25, activation=self._leak_relu) # 输入张量，神经元数量，激活函数，层id
        net = self._fc_layer(net, 4096, 26, activation=self._leak_relu)
        net = self._fc_layer(net, self.grid*self.grid*(self.class_num+5*self.box), 27) # 神经元数量7*7*(20+5*2)=7*7*30

        self.predicts = net

        if self.verbose:
            print("Finished build the network……")


    def _conv_layer(self, x, filter_size, stride, filter_num, layer_id ):
        """ 卷积层"""
        input_channel = x.get_shape().as_list()[-1] # 输入通道数
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, input_channel, filter_num], stddev=0.1)) # 权重节点
        bias = tf.Variable(tf.zeros([filter_num,])) # 偏置项
        # 手动padding
        pad_size = filter_size//2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)

        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
        conv_bias = tf.nn.bias_add(conv, bias)
        output = self._leak_relu(conv_bias)

        if self.verbose:
            print(" Layer %d : Type=Convolutional Layer, Filter_size=%d, Stride=%d, Filter_num=%d, Output_shape=%s"\
                    %(layer_id, filter_size, stride, filter_num, str(output.get_shape())))
        return output

    def _leak_relu(self, x, alpha= 0.1):
        """激活函数"""
        return tf.maximum(alpha*x, x)

    def _maxpool_layer(self, x, pool_size, stride, layer_id):
        """最大池化层"""
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1], [1, stride, stride, 1], padding="SAME")
        if self.verbose:
            print(" Layer %d : Type=Maxpool, Pool_size=%d, Stride=%d, Output_shape=%s"\
                    %(layer_id, pool_size, stride, str(output.get_shape())))
        return output

    def _flatten(self, x):
        """将x展开为一维"""
        x_trans = tf.transpose(x, [0, 3, 1, 2]) # 转置
        nums = np.product(x.get_shape().as_list()[1:]) # 列表元素全部相乘
        return tf.reshape(x_trans, [-1, nums])

    def _fc_layer(self, x, neuron_num, layer_id, activation = None):
        """全连接层"""
        input_num = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([input_num, neuron_num], stddev=0.1))
        bias = tf.Variable(tf.zeros([neuron_num,]))

        output = tf.nn.xw_plus_b(x, weight, bias) # 相当于matmul(x, weights) + biases

        if activation:
            output = activation(output)

        if self.verbose:
            print("Layer %d : Type=Full_connect_layer, Neuron_num=%d, output_shape=%s"\
                    %(layer_id, neuron_num, str(output.get_shape())))
        return output

    def _load_weights(self, weights_file):
        """加载权重文件"""

        if self.verbose:
            print("Start to laod the weights file:%s" %(weights_file))

        saver= tf.train.Saver()
        saver.restore(self.sess, weights_file)

    # -----------------------------------------------------------------------------------
    def detect_from_file(self, image, imshow=True, 
        detected_boxes_file='boxes_file.txt', detected_image_file='detected_image.jpg'):
        """检测图像"""

        image = cv2.imread(image) # 读入图像
        img_h, img_w, _ = image.shape  # 图片的高度和宽度
        predicts = self._detect_from_image(image) # 将图像输入卷积神经网络进行前向传播计算
        predict_boxes = self._interpret_predicts(predicts, img_h, img_w)  # 解析预测结果，得到检测框
        self.show_results(image, predict_boxes, imshow, detected_boxes_file, detected_image_file )

    def _detect_from_image(self, image):
        """ 将图像输入卷积神经网络进行前向传播计算"""
        img_resize = cv2.resize(image, (448, 448))  # 缩放图片，使用网络的输入尺寸要求
        img_RGB = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)  # 颜色空间转为RGB
        img_np = np.array(img_RGB)  
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32) 
        _images[0] = (img_np/255.0) * 2.0 - 1.0 # 图像转数组
        predicts = self.sess.run(self.predicts, feed_dict={self.images: _images})[0]
        return predicts

    def _interpret_predicts(self, predicts, img_h, img_w ):

        """解析预测结果，得到检测框"""
        idx1 = self.grid*self.grid*self.class_num  # 7*7* 20类
        idx2 = idx1 + self.grid*self.grid*self.box  # 7*7* 2预测框

        class_prob = np.reshape(predicts[:idx1], [self.grid, self.grid, self.class_num]) # 类别概率
        confidence = np.reshape(predicts[idx1:idx2], [self.grid, self.grid, self.box]) # 检测框置信度
        boxes = np.reshape(predicts[idx2:], [self.grid, self.grid, self.box, 4])  # 检测框坐标
        
        # 类别置信度分数
        scores = np.expand_dims(confidence, -1) * np.expand_dims(class_prob, 2) # 扩展维度再相乘 [7,7,2,1]*[7,7,1,20] = [7,7,2,20]
        scores = np.reshape(scores, [-1, self.class_num])  # [S*S*B, C] 7*7*2个预测框，每个20个类别预测值 [7*7*2,20]
        
        # 转换检测框中心的坐标
        boxes[:, :, :, 0] += self.x_offset
        boxes[:, :, :, 1] += self.y_offset
        boxes[:, :, :, :2] /= self.grid

        # 检测框的高和宽
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # 相对坐标转为绝对坐标(乘图像的高和宽)
        boxes[:, :, :, 0] *= img_w
        boxes[:, :, :, 1] *= img_h
        boxes[:, :, :, 2] *= img_w
        boxes[:, :, :, 3] *= img_h

        boxes = np.reshape(boxes, [-1, 4])

        # 将类别置信度分数过低的置为0
        scores[scores < self.conf_threshold] = 0.0
        # 非极大值抑制
        self._non_max_suppression(scores, boxes)


        predict_boxes = []  # (class, x, y, w, h, scores)
        max_idxs = np.argmax(scores, axis=1) # 每一个预测框的最大类别置信分数的索引 scores形状[7*7*2,30]
        for i in range(len(scores)): # 遍历每个预测框
            max_idx = max_idxs[i]
            if scores[i, max_idx] > 0.0:
                predict_boxes.append((self.classes[max_idx], boxes[i,0], boxes[i,1], boxes[i,2], boxes[i,3], scores[i, max_idx]))
        return predict_boxes

    def _non_max_suppression(self, scores, boxes):
        """非极大值抑制,针对每一类"""
        for c in range(self.class_num):
            sorted_idxs = np.argsort(scores[:, c]) # 排序后的索引
            last = len(sorted_idxs) - 1 # 取最大值用
            while last > 0:
                if scores[sorted_idxs[last], c] < 1e-6: # 类c的最大置信分数scores[sorted_idxs[last], c]
                    break
                for i in range(last): # 逐个取最大置信分数，计算iou，超过阈值置为0（如两辆车，各自有多个预测框）
                    if scores[sorted_idxs[i], c] < 1e-6:
                        continue
                    if self._iou(boxes[sorted_idxs[i]], boxes[sorted_idxs[last]]) > self.iou_threshold:
                        scores[sorted_idxs[i], c] = 0.0
                last -= 1 

    def _iou(self, box1, box2): # [x,y,w,h]
        """计算两个预测框的交并比"""
        inter_w = np.minimum(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
                np.maximum(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])

        inter_h = np.minimum(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
                np.maximum(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])

        if inter_h < 0 or inter_w < 0:
            inter_area = 0
        else:
            inter_area = inter_w * inter_h

        union_area = box1[2]*box1[3] + box2[2]*box2[3] - inter_area
        return inter_area/union_area

    def show_results(self, image, predict_boxes, imshow=True, detected_boxes_file=None,detected_image_file=None):
        """在图片上绘制检测框"""
        results = predict_boxes
        img_copy = image.copy()
        if detected_boxes_file:
            f = open(detected_boxes_file, 'w')
        # 遍历检测框，获取每个的信息
        for i in range(len(results)): # i [class,x,y,w,h,scores]
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2

            if self.verbose:
                print(" Class: %s, [x, y, w, h]=[%d, %d, %d, %d], Confidence=%f" % (results[i][0],x,y,w,h,results[i][-1]))
            cv2.rectangle(img_copy, (x-w, y-h), (x+w, y+h), (0,255,0), 2) # 画检测框
            cv2.rectangle(img_copy, (x-w,y-h-20), (x+w,y-h), (0,255,0), -1) # 画文本框
            cv2.putText(img_copy, results[i][0] + '%.2f' % results[i][5], (x-w+5,y-h-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1) # 照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
            if detected_boxes_file: # 将检测框信息写入文件
                f.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + str(results[i][5]) + '\n')
        if imshow:
            cv2.imshow('YOLO_small detection', img_copy) # 显示图片（窗口名称，图片）
            cv2.waitKey(0) # 等待毫秒数，0表示无限等待
        if detected_image_file: # 保存图片
            cv2.imwrite(detected_image_file, img_copy) # 参数：保存文件名，读入图片
        if detected_boxes_file: # 关闭检测框信息文件
            f.close() 

if __name__ == "__main__":
    yolo = YOLO("./weights/YOLO_small.ckpt")
    yolo.detect_from_file("./test/car.jpg")






























