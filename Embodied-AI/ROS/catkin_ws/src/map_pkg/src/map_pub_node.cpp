#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>

int main(int argc, char  *argv[])
{
    ros::init(argc, argv, "map_pub_node");
    ros::NodeHandle nh;
    ros::Publisher map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/map", 1);
    ros::Rate loop_rate(10);
    while(ros::ok())
    {
        nav_msgs::OccupancyGrid msg;
        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();
        //地图相对坐标系原点的偏移量
        msg.info.origin.position.x = 1.0; 
        msg.info.origin.position.y = 0.0;
        msg.info.resolution = 1.0; //栅格的长度
        msg.info.width = 4;
        msg.info.height = 2;

        msg.data.resize(msg.info.width * msg.info.height);
        msg.data[0]=90;
        msg.data[1]=100;
        msg.data[2]=0;
        msg.data[3]=-1;
        map_pub.publish(msg);
        loop_rate.sleep();

    }
    return 0;
}
