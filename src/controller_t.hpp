/*
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 15/08/2021

	[TODO: DESCRIPTION]
*/
#ifndef CONTROLLER
#define CONTROLLER 1

// Utilities
#include <iostream>

// Math
#include <Eigen/Dense>
#include <cmath>

using namespace std;

/*
	[TODO: DESCRIPTION]
*/
class controller_t {
	private:
		// Inverse Kinematics Variables
		// 
		double l_s_, l_w_; 
		double off_0_, off_1_;
		double h_1 = sqrt( pow(this->off_0_, 2) + pow( this-> off_1_, 2) );
		

	public:
		controller_t(void) {}
		/*
			Args:
				t: double  ->  Timestep.
				f_i: double  ->  i-th leg frequency offset (from NN policy).
				Hi_z: Eigen::Vector3d  ->  i-th leg horizontal frame z component.
				sigma_i_0: double  ->  i-th leg initial phase.
				h: double  ->  Maximun foot height. Set by default to 0.2.
				f_0: double  ->  Robot gait common frequency. Set by default to 1.25.
			Return:
				Eigen::Vector3d  ->  Vector expresed in the i-th leg horizontal frame Hi,
					representing de target foot position.
			References:
				* 	Learning Quadrupedal Locomotion over Challenging Terrain (2020).
					(p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
					https://arxiv.org/pdf/2010.11251.pdf
		*/
		Eigen::Vector3d FTG(
			double t,
			double f_i,
			Eigen::Vector3d Hi_z,
			double sigma_i_0,
			double h = 0.2,
			double f_0 = 1.25
		)
		{
			double sigma_i = (sigma_i_0 + (f_0 + f_i) * t) % (2 * M_PI);
			double k = 2 * (sigma_i - M_PI) / M_PI;

			if ( 0 < k  && k < 1)
			{
				return (h * ( -2* pow(k, 3) + 3*pow(k,2)) - 0.5) * Hi_z;
			}
			else if (1 < k  && k < 2)
			{
				return (h * ( 2* pow(k, 3) - 9* pow(k, 2) +  12*k - 4) - 0.5) * Hi_z;
			}
			else
			{
				return 0.5  * Hi_z;
			};
		}

		/*
			[TODO: DOCUMENTATION]
		*/
		Eigen::Vector3d inverse_kinematics(
								Eigen::Vector3d r
								)
		{ 	// Input
			double x = r[0], y = r[1], z = r[2];
			// Output
			double theta_h, theta_s, theta_w;


			double h_2 = sqrt( pow(z,2) + pow(y,2) );

			double r_o =  sqrt( pow(h_2,2) - pow(this->off_1_,2) ) - this->off_0_ ;
		
			theta_h = atan(y/z) - asin(this->off_1_ /h_2);

			double h = sqrt(pow(r_o,2) + pow(x,2));

			theta_s = acos((pow(h,2) + pow(this->l_s_,2) - pow(this->l_w_,2))/
					  (2*h*this->l_s_)) -asin(x/h);

			theta_w = acos((pow(this->l_w_,2) + pow(this->l_s_,2)-pow(h,2))/
					  (2*this->l_w_*this->l_s_));

			return (theta_h, theta_s, theta_w);
		};
};

#endif