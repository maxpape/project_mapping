/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include <cstdio>
#include <ros/ros.h>
#include <math.h>

// Services
#include "laser_assembler/AssembleScans.h"

// Messages
#include "sensor_msgs/PointCloud.h"
#include "sensor_msgs/PointCloud2.h"
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/JointState.h>



namespace laser_assembler
{

class PeriodicSnapshotter
{

public:

	PeriodicSnapshotter()
	{
	  	// create private nodehandle
		ros::NodeHandle n_("~");

		// create a publisher for the clouds that we assemble
		pub_ = n_.advertise<sensor_msgs::PointCloud2> ("assembled_cloud_2", 1);

		// subscribe to joint states to get motor position
		sub_ = n_.subscribe<sensor_msgs::JointState> ("/joint_states",40, &PeriodicSnapshotter::motorCallback, this);

		// create the service client for calling the assembler
		client_ = n_.serviceClient<AssembleScans>("/assemble_scans");

		// initialize min/max angles and indicator for if changes has happened
		// has_changed_min/max_ is needed, so service is only called once
		curr_angle_ = 0;
		max_angle_ = M_PI/2-0.1;
		min_angle_ = -M_PI/2+0.1;
		has_changed_min_ = false;
		has_changed_max_ = false;
	}

  void motorCallback (const sensor_msgs::JointStateConstPtr &motorMsg)
  {
    
	AssembleScans srv;

	// save current motor position
    curr_angle_ = motorMsg->position[0];
    //ROS_INFO("current angle: %f", curr_angle_) ;
    // when current angle is smaller then -pi/2 and start time has not been changed yet: pass
    if (((curr_angle_) < min_angle_) and (!has_changed_min_))
    {

    	// check has_changed_max_, if not set: take current time as start time
    	// if set: take current time as stop time
    	if (has_changed_max_)
    	{
    		stop_rotate_ = ros::Time::now();
    	}
    	else
    	{
    		start_rotate_ = ros::Time::now();
    	}
    	// set has_changed_min_ to true, after setting start or stop time
        has_changed_min_ = true;
    }

    // when current angle is greater then pi/2 and start time has not been changed yet: pass
    if (((curr_angle_) > max_angle_) and (!has_changed_max_))
    {

    	// check has_changed_min_, if not set: take current time as start time
    	// if set: take current time as stop time
    	if (has_changed_min_)
		{
			stop_rotate_ = ros::Time::now();
		}
		else
		{
			start_rotate_ = ros::Time::now();
		}
    	// set has_changed_max_ to true, after setting start or stop time
        has_changed_max_ = true;
    }


    // if start and stop time are both set, call service AssembleScans
    if (has_changed_min_ and has_changed_max_)
    {

    	// reset change indicators and set start/stop time for service call
    	has_changed_min_ = false;
    	has_changed_max_ = false;
		srv.request.begin = start_rotate_;
		srv.request.end   = stop_rotate_;

		if (client_.call(srv))
		{
			// print info to CLI
			// convert pointcloud from service call to pointcloud2 and publish it
			ROS_INFO("Published Cloud with %u points", (uint32_t)(srv.response.cloud.points.size())) ;
			cloud_ = srv.response.cloud;
			sensor_msgs::convertPointCloudToPointCloud2(cloud_, cloud2_);
			pub_.publish(cloud2_);
		}
    }

    return;
  }


private:

  ros::Publisher pub_;
  ros::Subscriber sub_;
  ros::ServiceClient client_;


  double curr_angle_;
  double max_angle_;
  double min_angle_;
  bool has_changed_min_;
  bool has_changed_max_;
  sensor_msgs::PointCloud cloud_;
  sensor_msgs::PointCloud2 cloud2_;
  ros::Time start_rotate_;
  ros::Time stop_rotate_;
} ;

}

using namespace laser_assembler ;

int main(int argc, char **argv)
{
    
  ROS_INFO("AAAAAAAAAAAAAAAAAAAAAA") ;
  ros::init(argc, argv, "periodic_snapshotter");
  ros::NodeHandle n;

  ROS_INFO("Waiting for [build_cloud] to be advertised");
  ros::service::waitForService("build_cloud");
  ROS_INFO("Found build_cloud! Starting the snapshotter");
  PeriodicSnapshotter snapshotter;
  ros::spin();
  return 0;
}
