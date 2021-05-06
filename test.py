import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer

#检测的类Detector
class Detector(object):

    def __init__(self, net, weight_file):   # Yolo网络
        self.net = net    #网络
        self.weights_file = weight_file  #保留模型的文件

        self.classes = cfg.CLASSES   # PASCAL VOC数据集的20个类别
        self.num_class = len(self.classes)  #20
        self.image_size = cfg.IMAGE_SIZE    #image_size 448
        self.cell_size = cfg.CELL_SIZE      #cell_size 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL  #每一个cell预测的框 2
        self.threshold = cfg.THRESHOLD    #0.2
        self.iou_threshold = cfg.IOU_THRESHOLD  #iou阈值 0.5
        self.boundary1 = self.cell_size * self.cell_size * self.num_class  #7*7*20
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell    #7*7*20 + 7*7*2

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())   #tensorflow中初始化全局变量

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)  #从模型文件中恢复会话

    def draw_result(self, img, result):   #输出结果
        print("hell")
        print(len(result))
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):  #检测
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))   #resize大小
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0        #对图片处理一下
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))  #reshape一下， [batch_size, image_size, image_size, 3]

        result = self.detect_from_cvmat(inputs)[0]  #返回已经是真实的坐标了

        print("输出result1:", result[0][1])
        print("输出result2:", result[0][2])
        print("输出result3:", result[0][3])
        print("输出result4:", result[0][4])

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)  #x_center,是真实的坐标了
            result[i][2] *= (1.0 * img_h / self.image_size)  #y_center
            result[i][3] *= (1.0 * img_w / self.image_size)  #width
            result[i][4] *= (1.0 * img_h / self.image_size)  #height

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})  #网络的输出
        print("net_output:",net_output.shape)  #其维度为[batch_size, 7*7*30]
        results = []
        for i in range(net_output.shape[0]):  #把每一张图片的预测放到一个list中
            results.append(self.interpret_output(net_output[i]))   #这一步是关键

        return results

    def interpret_output(self, output):  #进行阈值筛选（筛选的是类别置信度）和进行非极大值抑制
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))   #维度为[7,7,2,20]
        class_probs = np.reshape(   #reshape之后，其维度为[7, 7, 20]
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(    #reshape成[7, 7, 2]两个框的置信度
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(    #reshape成[7, 7, 2, 4]的坐标框
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.array(   #因为网络预测出来的是偏移量，因此要恢复
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size  #得到(x_center, y_cwenter)相对于每一张图片的位置比例，self.cell_size=7
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])     #得到预测的宽度和高度乘以平方才能得到相对于整张图片的比例，于loss function中的定义一致

        boxes *= self.image_size   #得到相对于原图的坐标框(这里的原图只resize之后的图，比如448*448)

        for i in range(self.boxes_per_cell):   #得到类别置信度， probs维度为[7, 7, 2, 20]
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        # filter_mat_probs 维度 [7, 7, 2, 20]
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')   #如果大于self.threshold，那么其对应的位置为true,反正false
        # np.nonzero()返回非0的index，类型为tuple，长度为len(dims), 比如filter_mat_probs的shape为(7, 7, 2, 20)，dims=4，
        # 则np.nonzero()返回len=4的tuple，boxes_filtered[0]表示第0维不为0的index，同理,boxes_filtered[1]表示第1维不为0的index；
        # tuple中的每个array都相等，因为每一个点都是由这4个坐标唯一确定的
        # filter_mat_boxes, type: tuple, len: 4
        filter_mat_boxes = np.nonzero(filter_mat_probs)    #找到为true的地方，用1来表示true, false是0

        # boxes_filtered shape: (num_filter_boxes, 4), 比如: (1546, 4), 表示一共有1546个类别置信度大于阈值的框（里面存在重复的，
        # 比如某一个框预测的猫的概率0.3，狗的概率0.4，猪的概率0.3...），第二个维度4表示box的位置
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]] #找到框的2位置，第三维度，不会用到第四维，因为第三维表示每个box的confidence，第四位表示类别的置信度
        probs_filtered = probs[filter_mat_probs]  #找到符合的类别置信度
        print('probs_filtered shape: ', probs_filtered.shape)

        # classes_num_filtered shape: (num_filter_boxes, ), 比如：（1546,)
        # 这里应该是probs，原文写的filter_mat_probs，凑巧对了
        classes_num_filtered = np.argmax(  #若该cell类别置信度大于阈值，则只取类别置信度最大的那个框，一个cell只负责预测一个类别
            probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        print('filter_mat_probs shape: ', filter_mat_probs.shape)
        print('filter_mat_probs content: ', filter_mat_probs)
        print('classes_num_filtered shape: ', classes_num_filtered.shape)
        print('classes_num_filtered content: ', classes_num_filtered)
        print('')

        argsort = np.array(np.argsort(probs_filtered))[::-1]  #类别置信度排序
        boxes_filtered = boxes_filtered[argsort]  #找到符合条件的框，从大到小排序
        probs_filtered = probs_filtered[argsort]  #找到符合条件的类别置信度，从大到小排序
        classes_num_filtered = classes_num_filtered[argsort]  #类别数过滤

        print('boxes_filtered shape: ', boxes_filtered.shape)
        print('probs_filtered shape: ', probs_filtered.shape)
        # TODO: 是否根据相同类别？
        for i in range(len(boxes_filtered)):  #非极大值抑制算法， iou_threshold=0.5
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]  #经过阈值和非极大值抑制之后得到的框
        probs_filtered = probs_filtered[filter_iou]  #经过阈值和非极大值抑制之后得到的类别置信度
        classes_num_filtered = classes_num_filtered[filter_iou]  #经过非极大值抑制之后得到的类别，一个cell只负责预测一个类别

        print('boxes_filtered shape: ', boxes_filtered.shape)
        print('probs_filtered shape: ', probs_filtered.shape)
        print('boxes_filtered content: ', boxes_filtered)
        print('probs_filtered content: ', probs_filtered)

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):  #计算iou的
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):     #相机检测
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):  #图片检测
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()   #记录检测开始的时间
        result = self.detect(image)   #检测
        detect_timer.toc()   #结束检测开始的时间
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)

#Detector的main函数
def main():
    parser = argparse.ArgumentParser()   #参数解析
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='./YOLO_small.ckpt', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)   #定义网络的框架
    # weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)  #模型文件路径
    weight_file = args.weight_dir
    detector = Detector(yolo, weight_file)   #初始化Detector类

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = 'test/street1.jpeg'  #测试文件
    detector.image_detector(imname)
    imname = 'test/car2.jpeg'
    detector.image_detector(imname)
    imname = 'test/car3.jpeg'
    detector.image_detector(imname)
    imname = 'test/person2.jpeg'
    detector.image_detector(imname)

#main函数
if __name__ == '__main__':
    main()  #调用main函数
