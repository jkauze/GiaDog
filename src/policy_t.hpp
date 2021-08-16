/*
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 15/08/2021

	[TODO: DESCRIPTION]
*/
#include <iostream>

using namespace std;

/*
	[TODO: DESCRIPTION]
*/
class policy_t {
	public:
		policy_t(void) {
			cerr << "\e[1;31mError.\e[0m Abstract class \e[1mpolicy_t\e[0m can't be "
				<< "instaciated.";
			exit(1);
		}

		/*
			[TODO: DESCRIPTION]
		*/
		virtual void compute(void) {
			cerr << "\e[1;31mError.\e[0m Not implemented method.\n";
			exit(1);
		 }
};

/*
	[TODO: DESCRIPTION]
*/
class ars_t : public policy_t {
	public:
		ars_t(void) { }

		/*
			[TODO: DOCUMENTATION]
		*/
		void compute(void) { 
			// [TODO: IMPLEMENTATION]
		 }
};
