/*
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 15/08/2021

	[TODO: DESCRIPTION]
*/
#pragma once
#ifndef CONTROLLER
#define CONTROLLER 1

// Utilities
#include <iostream>

// Math
#include <Eigen/Dense> //Matrix manipulation (numpy like arrays)
#include <cmath>


/*
	[TODO: DESCRIPTION]
*/
class controller_t {
	private:
		/*

			Inverse Kinematics variables
			
			l_s_   : double -> Lenght of the shoulder joint
			l_w_   : double -> Lenght of the wrist joint
			off_0_ : double -> Vertical offset betweet hip and shoulder link
			off_1_ : double -> Horizontal offset betweet hip and shoulder link
		
		*/
		
		double l_s_, l_w_, off_0_, off_1_; 
		
		/*

			Foot Trajectory Generator variables 
			
			Hi_z       : Eigen::Vector3d -> i-th leg horizontal frame z component.
			
			sigma_i_0_ : double -> Lenght of the wrist joint.
			
			h_   : double -> Maximun foot height. # Default 0.2 m
			f_0_ : double -> Robot gait common frequency. # Default 1.25 Hz
		
		*/

		Eigen::Vector3d Hi_z_;
		double sigma_i_0_, h_, f_0_;
		

	public:
		controller_t(void) {}
		/*
			Generates a vector in R^3 representing the desired foot position (end efector)
			in the H_i frame corresponding to the robots i-th leg horizontal frame below 
			its hip.
			
			Args:
				t  : double  ->  Timestep.
				f_i: double  ->  i-th leg frequency offset (from NN policy).
				
			Return:
				Eigen::Vector3d  ->  Vector expresed in the i-th leg horizontal frame Hi,
					representing de target foot position.
			References:
				* 	Learning Quadrupedal Locomotion over Challenging Terrain (Oct,2020).
					(p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
					https://arxiv.org/pdf/2010.11251.pdf
		*/
		Eigen::Vector3d FTG(
			double t,
			double f_i
			)
		{
			const double pi = 3.141592653589793;
			double sigma_i;
			sigma_i = fmod(this->sigma_i_0_ + (this->f_0_ + f_i) * t, 2 * pi);
			double k = 2 * (sigma_i - pi) / pi;

			if ( 0 < k  && k < 1)
			{
				return (this->h_*(-2* pow(k, 3) + 3*pow(k,2)) - 0.5) * this->Hi_z_;
			}
			else if (1 < k  && k < 2)
			{
				return (this->h_*(2* pow(k, 3) - 9* pow(k, 2) + 12*k - 4) - 0.5)
						*this->Hi_z_;
			}
			else
			{
				return - 0.5 * this->Hi_z_;
			};
		}

		/*
			For a robot foot, given a position r in its horizontal frame, the joint angles
			are calcultated. (Hip, Shoulder, Wrist) [The angles are in radians]
			
			Args:
				r  : Eigen::Vector3d  ->  Objective foot position in the H_i frame.
				
			Return:
				Eigen::Vector3d  ->  Leg joint angles to reach the objective foot 
									 position r. In the order: (Hip, Shoulder, Wrist).
									 The joint angles are expresed in radians.
				 
			References:
				* 	Adham Elarabawy (May, 2020).
					12-DOF Quadrupedal Robot: Inverse Kinematics
					https://www.adham-e.dev/pdf/IK_Model.pdf
		*/
		Eigen::Vector3d inverse_kinematics(
								Eigen::Vector3d r
								)
		{ 	
			// Input
			double x = r[0], y = r[1], z = r[2];
			// Output
			double theta_h, theta_s, theta_w;

			// Auxiliary Variables
			double h_2 = sqrt( pow(z,2) + pow(y,2) );
			double r_o =  sqrt( pow(h_2,2) - pow(this->off_1_,2) ) - this->off_0_ ;
			double h = sqrt(pow(r_o,2) + pow(x,2));


			// Joint angles calculation
			
			theta_h = atan(y/z) - asin(this->off_1_ /h_2);

			theta_s = acos((pow(h,2) + pow(this->l_s_,2) - pow(this->l_w_,2))/
					  (2*h*this->l_s_)) -asin(x/h);

			theta_w = acos((pow(this->l_w_,2) + pow(this->l_s_,2)-pow(h,2))/
					  (2*this->l_w_*this->l_s_));

			return Eigen::Vector3d(theta_h, theta_s, theta_w);
		};
};

#endif