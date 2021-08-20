/*
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 15/08/2021

	[TODO: DESCRIPTION]
*/
#pragma once 

// Utilities
#include <iostream>
#include "utils.hpp"

// Maths
#include "Eigen/Dense"
#include <math.h>

// Velocity threshold
#define V_TH 0.6

/*
	[TODO: DESCRIPTION]
*/
class policy_t {
	public:
		policy_t(void) {
			error("Abstract class \e[1mpolicy_t\e[0m can't be instaciated.");
		}

		/*
			[TODO: DESCRIPTION]
		*/
		virtual void compute(void) {
			error("Not implemented method.");
		 }
};

/*
	[TODO: DESCRIPTION]
*/
class teacher_policy_t : public policy_t {
	private:
		// Command parameters.
		// Target horizontal direction
		Eigen::Vector2d v_command_;
		// Turning direction.
		int8_t w_;

	public:
		teacher_policy_t(double psi, int8_t w);

		/*
			Change the vector command.

			Args:
				psi: double  ->  Yaw angle to command direction in the base frame.
				w: int8_t  ->  Truning direction. Must be -1, 0 or 1. 1 refers to 
					counter-clockwise.
		*/
		void change_command(double psi, int8_t w);

		/*
			Reward function.

			Args:
				v: Eigen::Vector2d  ->  Base linear velocity.
				w: Eigen::Vector3d  ->  Base angular velocity.
				phi: double[4]  ->  FTG frequencies.
				h_scan: std::vector<double>[4]  ->  Height scan around each foot.
				t_contact: bool[4]  ->  Thigh contact states.
				s_contact: bool[4]  ->  Shank contact states.
				r_fd: Eigen::Vector3d[3][4]  ->  Foot target history at times t, t-1 
					and t-2.
				theta: std::vector<double>  ->  Joints positions.
			Return:
				double  ->  Reward value.
		*/
		double reward(
			Eigen::Vector3d v, 
			Eigen::Vector3d w, 
			double phi[4], 
			std::vector<double> h_scan[4],
			bool t_contact[4],
			bool s_contact[4],
			Eigen::Vector3d r_fd[3][4],
			std::vector<double> theta
		);
};



teacher_policy_t::teacher_policy_t(double psi, int8_t w) 
{
	this->change_command(psi, w);
}

void teacher_policy_t::change_command(double psi, int8_t w) 
{
	// Verificamos que los parametros tienen valores validos.
	if (w < -1 || w > 1) error("Turning direction w must be -1, 0 or 1.");

	this->v_command_(cos(psi), sin(psi));
	this->w_ = w;
}

double teacher_policy_t::reward(
	Eigen::Vector3d v, 
	Eigen::Vector3d w, 
	double phi[4], 
	std::vector<double> h_scan[4],
	bool t_contact[4],
	bool s_contact[4],
	Eigen::Vector3d r_fd[3][4],
	std::vector<double> theta
) {
	// Zero command
	bool zero = ! (this->v_command_[0] || this->v_command_[1]);

	// Base horizonal linear velocity.
	Eigen::Vector2d v_xy(v[0], v[1]);
	// Base horizontal linear velocity projected onto the command direction.
	double v_pr = v_xy.dot(this->v_command_);
	// Velocity orthogonal to the target direction.
	double v_0 = zero ? v_0 = (v_xy - v_pr * this->v_command_).norm() : v_0 = v.norm();

	// Base angular velocity Z.
	double w_z = w[2];
	// Base horizontal angular velocity.
	Eigen::Vector2d w_xy(w[0], w[1]);
	// Base angular velocity Z projected onto desired angular velocity.
	double w_pr = w_z * this->w_;

	// Set of such collision-free feet and index set of swing legs
	double f_clear = 0, i_swing = 0;
	// For every foot.
	for (int i = 0; i < 4; i++)
	{
		// If i-th foot is in swign phase.
		if (phi[i] >= M_PI) //Note: This may cause trouble: rec-> change for pi value
		{
			i_swing++;

			// Verify that r_f,i > max(H_scan,i)
			bool f_clear_i = true;
			for (double h : h_scan[i])
			{
				// r_fd at time t (0), foot i, position Z (2).
				if (r_fd[0][i][2] <= h)
				{
					f_clear_i = false;
					break;
				}
			}

			if (f_clear_i) f_clear++;
		}
	}


	/* =============== REWARDS =============== */
	// Linear Velocity Reward
	double r_lv;
	if (zero) r_lv = 0;
	else if (v_pr < V_TH) r_lv = exp(-2.0 * pow(v_pr - V_TH, 2));
	else r_lv = 1;

	// Angular Velocity Reward
	double r_av = 0;
	if (this->w_)
	{
		if (w_pr < V_TH) r_av = exp(-1.5 * pow(w_pr - V_TH, 2));
		else r_av = 1;
	}

	// Base Motion Reward
	double r_b = exp(-1.5 * pow(v_0, 2)) + exp(-1.5 * w_xy.dot(w_xy));

	// Foot Clearance Reward
	double r_fc = i_swing ? r_fc / i_swing : 1;

	// Body Collision Reward
	double r_bc = 0;
	for (bool contact : t_contact) r_bc -= contact ? 1 : 0;
	for (bool contact : s_contact) r_bc -= contact ? 1 : 0;

	// Target Smoothness Reward
	Eigen::Vector<double, 12> r_fd_tm0 = {
		r_fd[0][0][0], r_fd[0][0][1], r_fd[0][0][2],
		r_fd[0][1][0], r_fd[0][1][1], r_fd[0][1][2],
		r_fd[0][2][0], r_fd[0][2][1], r_fd[0][2][2],
		r_fd[0][3][0], r_fd[0][3][1], r_fd[0][3][2]
	};
	Eigen::Vector<double, 12> r_fd_tm1 = {
		r_fd[1][0][0], r_fd[1][0][1], r_fd[1][0][2],
		r_fd[1][1][0], r_fd[1][1][1], r_fd[1][1][2],
		r_fd[1][2][0], r_fd[1][2][1], r_fd[1][2][2],
		r_fd[1][3][0], r_fd[1][3][1], r_fd[1][3][2]
	};
	Eigen::Vector<double, 12> r_fd_tm2 = {
		r_fd[2][0][0], r_fd[2][0][1], r_fd[2][0][2],
		r_fd[2][1][0], r_fd[2][1][1], r_fd[2][1][2],
		r_fd[2][2][0], r_fd[2][2][1], r_fd[2][2][2],
		r_fd[2][3][0], r_fd[2][3][1], r_fd[2][3][2]
	};
	double r_s = - (r_fd_tm0 - 2.0 * r_fd_tm1 + r_fd_tm2).norm();

	// Torque Reward
	double r_tau = 0;
	for (double t : theta) r_tau -= abs(t);


	return (5*r_lv + 5*r_av + 4*r_b + r_fc + 2*r_bc + 2.5*r_s) / 100.0 + 2e-5 * r_tau;
}