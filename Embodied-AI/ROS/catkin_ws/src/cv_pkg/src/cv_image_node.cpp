#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
//包含 cv_bridge 头文件，用于在 ROS 图像消息（sensor_msgs/Image）和 OpenCV 图像格式（cv::Mat）之间进行转换。
#include<sensor_msgs/image_encodings.h>
//包含 sensor_msgs 中的图像编码格式定义，例如 BGR8、RGB8 等
#include<opencv2/imgproc/imgproc.hpp>
//包含 OpenCV 的图像处理模块头文件，提供图像处理功能，如图像滤波、边缘检测等
#include<opencv2/highgui/highgui.hpp>
//包含 OpenCV 的高级用户界面模块头文件，提供图像显示和窗口管理功能。
using namespace cv;
//使用 OpenCV 的命名空间 cv，避免在代码中多次使用 cv:: 前缀。
void Cam_RGB_Callback(const sensor_msgs::Image msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {   //将 ROS 的图像消息（sensor_msgs::Image）转换为 OpenCV 的图像格式（cv::Mat）。
        //BGR8 是一种图像编码格式，表示每个像素由 3 个字节组成，分别对应蓝（Blue）、绿（Green）、红（Red）三个颜色通道，每个通道 8 位（1 字节）
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", e.what());
        return;
    }
    //cv_bridge::CvImagePtr 对象中获取 OpenCV 的 cv::Mat 图像数据
    Mat imageOriginal=cv_ptr->image;
    //作用：在窗口中显示图像。
    // "RGB_Image"：窗口名称。
    // imageOriginal：要显示的图像数据。
    imshow("RGB_Image", imageOriginal);
    //作用：确保图像窗口能够有足够时间显示。
    waitKey(3);
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "cv_image_node");
    ros::NodeHandle nh;
    //kinect2含义：表示这是 Kinect 2 相机的数据话题。
    //qhd含义：表示 Quarter HD（四分之一高清）分辨率。
    ros::Subscriber rgb_sub = nh.subscribe("/kinect2/qhd/image_color_rect", 1, Cam_RGB_Callback);
    namedWindow("RGB_Image");

    ros::spin();
}