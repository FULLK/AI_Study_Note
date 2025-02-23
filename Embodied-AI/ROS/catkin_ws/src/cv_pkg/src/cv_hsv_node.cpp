#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/image_encodings.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

static int iLowH = 10;
static int iHighH = 40;

static int iLowS = 90; 
static int iHighS = 255;

static int iLowV = 1;
static int iHighV = 255;

void Cam_RGB_Callback(const sensor_msgs::Image msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr= cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    Mat imgOriginal = cv_ptr->image;


    //创建一个 cv::Mat 对象 imgHSV，用于存储转换后的 HSV 图像
    Mat imgHSV;
    //将原始图像 imgOriginal 从 BGR 颜色空间转换为 HSV 颜色空间，并存储到 imgHSV 中。
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
    //创建一个 std::vector<cv::Mat> 对象 hsvSplit，用于存储 HSV 图像的三个通道（H、S、V）。
    vector<Mat> hsvSplit;
    //将 imgHSV 图像的三个通道（H、S、V）分离，并存储到 hsvSplit 中
    split(imgHSV, hsvSplit);
    //equalizeHist 是 OpenCV 中用于对单通道图像进行直方图均衡化（Histogram Equalization）的函数。
    //对 hsvSplit[2]（V 通道，即亮度通道）进行直方图均衡化，并将结果存储回 hsvSplit[2] 
    //将输入图像的直方图拉伸为均匀分布，从而增强图像的对比度。
    equalizeHist(hsvSplit[2], hsvSplit[2]);
    //将多个单通道图像合并为一个多通道图像的函数。
    //将处理后的 HSV 通道（H、S、V）重新合并为一个三通道图像，并存储到 imgHSV 中。
    merge(hsvSplit, imgHSV);

    Mat imgThresholded;
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
    //创建一个 5x5 的矩形结构元素（kernel）。
    // MORPH_RECT：表示矩形形状。
    // Size(5, 5)：表示结构元素的大小为 5x5
    Mat element=getStructuringElement(MORPH_RECT, Size(5, 5));
    // 对 imgThresholded 图像进行 开运算。
    // 开运算：先腐蚀后膨胀。开运算：白色噪声被去除，图像的亮区域变得更加平滑。
    // 腐蚀：去除图像中的小亮斑（白色噪声）。
    // 膨胀：恢复图像中的主要亮区域。
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    // 作用：对 imgThresholded 图像进行 闭运算。
    // 闭运算：先膨胀后腐蚀。闭运算：黑色噪声被填补，图像的暗区域变得更加平滑。
    // 膨胀：填补图像中的小暗斑（黑色噪声）。
    // 腐蚀：恢复图像中的主要暗区域。
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

    int nTargetX = 0;
    int nTargetY = 0;
    int nPixCount = 0;
    int nImgWidth = imgThresholded.cols;
    int nImgHeight = imgThresholded.rows;
    int nImgChannels = imgThresholded.channels();
    for(int i=0; i<nImgHeight; i++)
    {
        for(int j=0; j<nImgWidth; j++)
        {
            if(imgThresholded.data[i*nImgWidth+j] == 255)
            {
                nTargetX += j;
                nTargetY += i;
                nPixCount++;
            }
        }
    }

    if(nPixCount > 0)
    {
        nTargetX /= nPixCount; 
        nTargetY /= nPixCount;
        printf("颜色质心坐标（ %d , %d）点数 = %d \n", nTargetX, nTargetY,nPixCount);

        //line_begin：定义了第一条线段的起点，坐标为 (nTargetX-10, nTargetY)。
        //line_end：定义了第一条线段的终点，坐标为 (nTargetX+10, nTargetY)。
        Point line_begin = Point(nTargetX-10, nTargetY);
        Point line_end = Point(nTargetX+10, nTargetY);
        //imgOriginal 图像上绘制一条从 line_begin 到 line_end 的红色线段。
        line(imgOriginal, line_begin, line_end, Scalar(0, 0, 255));
        //两点定义了一条垂直线段，长度为 20 个像素
        line_begin.x = nTargetX;
        line_begin.y = nTargetY-10;
        line_end.x = nTargetX;
        line_end.y = nTargetY+10;
        line(imgOriginal, line_begin, line_end, Scalar(0, 0, 255));
    }
    else
    {
        printf("未检测到目标！\n");
    }
    imshow("RGB", imgOriginal);
    imshow("HSV", imgHSV);
    imshow("Result", imgThresholded);
    cv::waitKey(5);
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "cv_hsv_node");
    ros::NodeHandle nh;
    ros::Subscriber rgb_sub = nh.subscribe("kinect2/qhd/image_color_rect", 1, Cam_RGB_Callback);
    namedWindow("Threshold", WINDOW_AUTOSIZE);
    createTrackbar("LowH", "Threshold", &iLowH, 179); //Hue (0 - 179) 缩小一半
    createTrackbar("HighH", "Threshold", &iHighH, 179);
    createTrackbar("LowS", "Threshold", &iLowS, 255);//Saturation (0 - 255)
    createTrackbar("HighS", "Threshold", &iHighS, 255);
    createTrackbar("LowV", "Threshold", &iLowV, 255);//Value (0 - 255)
    createTrackbar("HighV", "Threshold", &iHighV, 255);

    namedWindow("RGB");
    namedWindow("HSV");
    namedWindow("Result");

    ros::Rate loop_rate(30);
    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}