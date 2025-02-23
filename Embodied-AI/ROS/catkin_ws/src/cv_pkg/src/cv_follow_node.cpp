#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/image_encodings.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<geometry_msgs/Twist.h>

using namespace cv;
using namespace std;


static int iLowH = 10;
static int iHighH = 40;

static int iLowS = 90; 
static int iHighS = 255;

static int iLowV = 1;
static int iHighV = 255;

geometry_msgs::Twist vel_cmd;
ros::Publisher vel_pub;


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

    Mat imgHSV;
    vector<Mat> hsvSplit;
    cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

    split(imgHSV, hsvSplit);
    equalizeHist(hsvSplit[2], hsvSplit[2]);
    merge(hsvSplit, imgHSV);

    Mat imgThresholded;
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element); 
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
        printf("颜色质心坐标（ %d , %d）点数 = %d \n", nTargetX, nTargetY,nPixCount);nImgWidth;
        Point line_begin = Point(nTargetX-10,nTargetY);
        Point line_end = Point(nTargetX+10,nTargetY);
        line(imgOriginal,line_begin,line_end,Scalar(255,0,0));
        line_begin.x = nTargetX; line_begin.y = nTargetY-10; 
        line_end.x = nTargetX; line_end.y = nTargetY+10; 
        line(imgOriginal,line_begin,line_end,Scalar(255,0,0));

        float fVeFoward=(nImgHeight/2-nTargetY)*0.002; //差值越大，移动速度越大
        float fVelTurn=(nImgWidth/2-nTargetX)*0.002;  //差值越大，旋转速度越大
        vel_cmd.linear.x = fVeFoward;
        vel_cmd.linear.y = 0;
        vel_cmd.linear.z = 0;
        vel_cmd.angular.x = 0;
        vel_cmd.angular.y = 0;
        vel_cmd.angular.z= fVelTurn;
    }
    else
    {
        printf("未检测到颜色\n");
        vel_cmd.linear.x = 0;
        vel_cmd.linear.y = 0;
        vel_cmd.linear.z = 0;
        vel_cmd.angular.x = 0;
        vel_cmd.angular.y = 0;
        vel_cmd.angular.z= 0;
    }
    vel_pub.publish(vel_cmd);
    printf("移动速度：%f,%f,%f 旋转速度：%f,%f,%f\n",vel_cmd.linear.x,vel_cmd.linear.y,vel_cmd.linear.z,vel_cmd.angular.x,vel_cmd.angular.y,vel_cmd.angular.z);
    imshow("RGB", imgOriginal);
    imshow("Result", imgThresholded);
    cv::waitKey(1);
}
int main(int argc, char** argv)
{   
    ros::init(argc, argv, "cv_follow_node");
    ros::NodeHandle nh;
    ros::Subscriber rgb_sub = nh.subscribe("kinect2/qhd/image_color_rect", 1, Cam_RGB_Callback); 
    vel_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    namedWindow("Threshold", WINDOW_AUTOSIZE);

    createTrackbar("LowH", "Threshold", &iLowH, 179); //Hue (0 - 179) 缩小一半
    createTrackbar("HighH", "Threshold", &iHighH, 179);
    createTrackbar("LowS", "Threshold", &iLowS, 255);//Saturation (0 - 255)
    createTrackbar("HighS", "Threshold", &iHighS, 255);
    createTrackbar("LowV", "Threshold", &iLowV, 255);//Value (0 - 255)
    createTrackbar("HighV", "Threshold", &iHighV, 255);

    namedWindow("RGB");
    namedWindow("Result");

    ros::spin();


}