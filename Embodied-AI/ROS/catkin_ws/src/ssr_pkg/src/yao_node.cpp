#include<ros/ros.h>
#include<std_msgs/String.h>
int main(int argc, char  *argv[])
{   
    ros::init(argc,argv,"yao_node");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<std_msgs::String>("yao",10);

    ros::Rate loop_rate(10);
    while(ros::ok())//使用ture时候运行节点不能响应外界信号
    {
        printf("keep\n");
        std_msgs::String msg;
        msg.data = "do this all day woman!!!";
        pub.publish(msg);
        loop_rate.sleep();
    }
    printf("hello world");
    return 0;
}
