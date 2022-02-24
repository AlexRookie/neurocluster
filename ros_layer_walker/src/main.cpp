#include "RequesterSimple.hpp"
#include "Subscriber.hpp"
#include "timer.hpp"
#include "json.hpp"

#include <signal.h>
#include <chrono>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <ros_layer_walker/wheels_current.h>
#include <geometry_msgs/Twist.h>

#include <fstream>
using namespace std;
using json = nlohmann::json;

enum WALKER_NODE{DESELECT = 0, FRONT_LEFT = 1, FRONT_RIGHT = 2, REAR_RIGHT = 3, REAR_LEFT = 4};

static volatile int terminating = 0;

double xx, yy, thth, omom, vv;

ZMQCommon::RequesterSimple req("tcp://192.168.3.2:5601"); 
ZMQCommon::Subscriber sub;

void intHandler(int dummy) {
	terminating = 1;
}

double timeNow() {
  using chrono::duration_cast;
  using chrono::milliseconds;
  using chrono::system_clock;
  return (long long int)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()/1000.;
}


bool sendRequest(ZMQCommon::RequesterSimple & req, json const & jobj) {
  string in_msg = jobj.dump();
  string out_msg;
  ZMQCommon::RequesterSimple::status_t status;
  req.request(in_msg, out_msg, status);
//  cout << out_msg << endl;
  return status==ZMQCommon::RequesterSimple::status_t::STATUS_OK; // TODO: check ACK
}

bool powerEnable(ZMQCommon::RequesterSimple & req, bool enable) {
  json jobj;
  jobj["cmd"] = string("set_power_enable");
  jobj["enable"] = enable;
  return sendRequest(req, jobj);
}

bool setDeviceMode(ZMQCommon::RequesterSimple & req, int deviceMode) {
  json jobj;
  jobj["cmd"] = string("set_device_mode");
  jobj["device_mode"] = deviceMode;
  return sendRequest(req, jobj);
}

bool powerOn(ZMQCommon::RequesterSimple & req) {
  if (!powerEnable(req, true)) return false;
  if (!setDeviceMode(req, 5)) return false;
  return true;
}

bool walkerStop(ZMQCommon::RequesterSimple & req) {
  json jobj;
  jobj["cmd"] = string("walker_stop");
  return sendRequest(req, jobj);
}


bool powerOff(ZMQCommon::RequesterSimple & req) {
  return walkerStop(req) && powerEnable(req, false);
}

bool currentSteering(ZMQCommon::RequesterSimple & req, double left, double right, double acceleration) {
  json j_req;
  j_req["cmd"] = std::string("set_single_node_current");
  j_req["dest"] = WALKER_NODE::REAR_LEFT;
  j_req["value"] = acceleration;
  sendRequest(req, j_req);
  j_req["cmd"] = std::string("set_single_node_current");
  j_req["dest"] = WALKER_NODE::REAR_RIGHT;
  j_req["value"] = -acceleration;
  sendRequest(req, j_req);

  json jobj;
  jobj["cmd"] = string("set_front_current");
  jobj["right"] = right;
  jobj["left"] = left;

  return sendRequest(req, jobj);
}


void sub_callback(const char *topic, const char *buf, size_t size, void *data){
    json j;
    try{
	j = json::parse(string(buf, size));
	//j = j.at("state");
	double x = round((double)j.at("state").at("x")*100)/100.0;
	double y = round((double)j.at("state").at("y")*100)/100.0;
	//int z = j.at("z");
	double theta = j.at("state").at("theta");
	double omega = j.at("state").at("omega");
	double v = j.at("state").at("v");
	
	xx = x;
	yy = y;
	thth = theta;
	omom = omega;
	vv = v;

	/*double covxx = j.at("covariance")[0];
	double covxy = j.at("covariance")[1];
	double covyy = j.at("covariance")[2];
	double tmp[4] = {covxx,covxy,covxy,covyy};

	QGenericMatrix<2, 2, double> cov(tmp);

	cout << j.dump() << endl << endl;*/
	

    }catch(exception &e){
	cerr << "error parsing: " << e.what() << endl;
    }
}

void odometry(ros::Publisher odom_pub, tf::TransformBroadcaster odom_broadcaster, ros::Time current_time)
{
	double x = xx;
	double y = yy;
	double th = thth;

	double vx = vv*cos(th);
	double vy = vv*sin(th);
	double vth = omom;

	//cout << "x: " << x << " y: " << y << " theta: " << th << endl;

	//since all odometry is 6DOF we'll need a quaternion created from yaw
	geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(th);

	//first, we'll publish the transform over tf
	geometry_msgs::TransformStamped odom_trans;
	odom_trans.header.stamp = current_time;
	odom_trans.header.frame_id = "odom";
	odom_trans.child_frame_id = "base_link";

	odom_trans.transform.translation.x = x;
	odom_trans.transform.translation.y = y;
	odom_trans.transform.translation.z = 0.0;
	odom_trans.transform.rotation = odom_quat;

	//send the transform
	odom_broadcaster.sendTransform(odom_trans);

	//next, we'll publish the odometry message over ROS
	nav_msgs::Odometry odom;
	odom.header.stamp = current_time;
	odom.header.frame_id = "odom";
	odom.child_frame_id = "base_link";

	//set the position
	odom.pose.pose.position.x = x;
	odom.pose.pose.position.y = y;
	odom.pose.pose.position.z = 0.0;
	odom.pose.pose.orientation = odom_quat;

	//set the velocity
	odom.twist.twist.linear.x = vx;
	odom.twist.twist.linear.y = vy;
	odom.twist.twist.angular.z = vth;

	//publish the message
	odom_pub.publish(odom);
}

void command_torque(const geometry_msgs::Twist &curr)
{
	currentSteering(req,curr.linear.x,curr.linear.y,curr.linear.z);
}


int main (int argc, char *argv[])
{  
  ZMQCommon::RequesterSimple arucolist("tcp://192.168.3.1:5567");
  ZMQCommon::RequesterSimple::status_t status;
  string response;

  ifstream ifs("/Users/placido/catkin_ws/src/ros_layer_walker/listaQR.json");
  json jobj_aruco = json::parse(ifs);
  cout << sendRequest(arucolist,jobj_aruco) << endl;
  arucolist.close();
  
  sub.register_callback([&](const char *topic, const char *buf, size_t size, void *data){sub_callback(topic,buf,size,data);});
  sub.start("tcp://192.168.3.1:5563","LOC");

  while (!terminating && !powerOn(req)) {
    cerr << "[WARN] Cannot start robot..." << endl;
    this_thread::sleep_for(chrono::milliseconds(100));
  }

  if (terminating) return 0;

  ros::init(argc, argv, "ros_layer_walker");

  ros::NodeHandle n;
  ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("odom", 50);
  tf::TransformBroadcaster odom_broadcaster;

  ros::Subscriber ros_sub = n.subscribe("cmd_vel", 1000, command_torque);

  ros::Time current_time;

  ros::Rate r(20.0);
  while(n.ok() && !terminating){
    ros::spinOnce();
    current_time = ros::Time::now();
    
    odometry(odom_pub,odom_broadcaster,current_time);

    r.sleep();
  }
  
  sub.stop();
  walkerStop(req);
  powerOff(req);
  req.close();
      
  return 0;
}
