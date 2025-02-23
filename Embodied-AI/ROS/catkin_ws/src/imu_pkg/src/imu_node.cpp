#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf/tf.h> // 四元数转换为欧拉角
#include <geometry_msgs/Twist.h>
/**
 * IMUCallback 函数用于处理IMU数据回调。
 * 该函数接收IMU消息，检查方向估计的有效性，并将四元数转换为欧拉角后打印出来。
 *
 * @param msg 接收到的IMU消息，包含四元数和其他传感器数据。
 */
ros::Publisher vel_pub;
void IMUCallback(sensor_msgs::Imu msg)
{
    // 检查是否有有效的方向估计值
    if (msg.orientation_covariance[0] < 0) 
        return;

    // 创建 tf::Quaternion 对象，用于后续将四元数转换为欧拉角
    tf::Quaternion quaternion(
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w
    );

    // 将四元数转换为欧拉角（roll, pitch, yaw）
    double roll, pitch, yaw;
    tf::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
    // tf::Matrix3x3 是一个 3x3 的矩阵类，用于表示旋转矩阵。
    // tf::Matrix3x3(quaternion)：将四元数 quaternion 转换为旋转矩阵。

    // getRPY 方法从旋转矩阵中提取欧拉角（roll、pitch、yaw）。
    // 将欧拉角从弧度转换为角度 将弧度乘以π/180，即可将弧度转换为角度。
    roll = roll * 180 / M_PI;
    pitch = pitch * 180 / M_PI;
    yaw = yaw * 180 / M_PI;

    // 打印转换后的欧拉角信息
    ROS_INFO("roll滚转: %.2f, pitch俯仰: %.2f, yaw朝向: %.2f", roll, pitch, yaw);

    double target_yaw = 90;
    double diff_angle=target_yaw-yaw;
    geometry_msgs::Twist vel_cmd;
    vel_cmd.angular.z=diff_angle*0.01;
    vel_cmd.linear.x=0.1;
    vel_pub.publish(vel_cmd);

}

int main(int argc, char *argv[])
{
    // 设置区域，以便正确显示字符
    setlocale(LC_ALL, "");

    // 初始化ROS节点
    ros::init(argc, argv, "imu_node");

    // 创建节点句柄
    ros::NodeHandle n;

    // 订阅IMU数据
    ros::Subscriber imu_sub = n.subscribe("/imu/data", 10, IMUCallback);

    vel_pub=n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

    // 进入消息循环，等待回调处理
    ros::spin();

    // 返回成功
    return 0;
}