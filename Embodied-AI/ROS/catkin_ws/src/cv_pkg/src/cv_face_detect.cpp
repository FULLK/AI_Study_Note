#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/image_encodings.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;


static CascadeClassifier face_cascade;//cv::CascadeClassifier 对象，用于加载 Haar 特征分类器文件。

static Mat frame_gray; //存储黑白图像
static vector<Rect> faces;

static vector<Rect>::const_iterator face_iter;

void CallbackRGB(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch( cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    Mat imgOriginal = cv_ptr->image;
    //将原始图像转换为灰度图像，因为人脸检测通常在灰度图像上进行
    cvtColor(imgOriginal, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //detectMultiScale：CascadeClassifier 类的成员函数，用于检测图像中的多尺度特征（如人脸)
    // frame_gray：输入的灰度图像。
    // faces：输出的矩形框数组，表示检测到的人脸区域。
    // 1.1：尺度因子，表示每次图像缩放的比例。
    // 2：最小邻居数，表示每个候选区域的最小邻居数。
    // 0|CASCADE_SCALE_IMAGE：检测选项，包括图像缩放。
    // Size(30, 30)：最小检测窗口大小，表示检测到的人脸的最小尺寸。
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

    if(faces.size()>0)
    {   
        //遍历检测到的所有人脸区域。
        for(face_iter = faces.begin(); face_iter != faces.end(); ++face_iter)
        {   
            // imgOriginal：原始图像，用于绘制矩形框。
            // Point(face_iter->x, face_iter->y)：矩形框的左上角点。
            // Point(face_iter->x + face_iter->width, face_iter->y + face_iter->height)：矩形框的右下角点。
            // CV_RGB(255, 0, 255)：矩形框的颜色，表示为 RGB 值，这里是紫色。
            // 2：矩形框的边框厚度。
            rectangle(
                imgOriginal, 
                Point(face_iter->x , face_iter->y), 
                Point(face_iter->x+face_iter->width, face_iter->y+face_iter->height), 
                CV_RGB(255, 0, 255),
                2);
        }
    }
    imshow("face", imgOriginal);
    waitKey(1);
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "cv_face_detect");
    namedWindow("face");
    
    std::string strLoadFile;
    char const* home = getenv("HOME");
    strLoadFile = home;
    strLoadFile += "/AI_Study_Note/Embodied-AI/ROS/catkin_ws/src";
    //haarcascade_frontalface_alt.xml 文件是一个 XML 文件，包含了用于人脸检测的 Haar 特征分类器的模型参数
    strLoadFile += "/wpr_simulation/config/haarcascade_frontalface_alt.xml";

    bool res=face_cascade.load(strLoadFile);
    if(res==false)
    {
        ROS_ERROR("Load haarcascade_frontalface_alt.xml failed!");
        return 0;
    }
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("/kinect2/qhd/image_color_rect", 1, CallbackRGB);
    ros::spin();
    return 0;
}