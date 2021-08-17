#include "terrain_t.hpp"
#include "utils.hpp"

void syntax_error(void);

int main(int argc, char **argv) {
	if (argc < 3) syntax_error();

	terrain_t t;

	if ((std::string) argv[1] == "--hills")
	{
		if (argc != 7) syntax_error();
		t.adaptative_hills(
			atof(argv[2]), 
			atof(argv[3]), 
			atof(argv[4]),
			atoi(argv[5])
		);
		t.save((std::string) argv[6]);
	}
	else if ((std::string) argv[1] == "--maincra")
	{
		if (argc != 6) syntax_error();
		t.adaptative_maincra(
			atoi(argv[2]), 
			atof(argv[3]), 
			atoi(argv[4])
		);
		t.save((std::string) argv[4]);
	}
	else if ((std::string) argv[1] == "--stairs")
	{
		if (argc != 5) syntax_error();
		t.adaptative_stairs(
			atoi(argv[2]), 
			atoi(argv[3])
		);
		t.save((std::string) argv[4]);
	}
	else if ((std::string) argv[1] == "--run")
	{
		if (argc != 3) syntax_error();
		terrain_t t((std::string) argv[2]);
	}
	else 
	{
		syntax_error();
	}

	return 0;
}

void syntax_error(void) {
	error(
		"Invalid syntax. Use:\n\n"
		"  \e[1mterrain_gen --hills\e[0m [\e[3;4mROUGHNESS\e[0m] [\e[3;4mFREQUENCY\e[0m] "
					"[\e[3;4mHEIGHT\e[0m] [\e[3;4mSEED\e[0m] [\e[3;4mFILE_OUT\e[0m] \n"
		"  \e[1mterrain_gen --maincra\e[0m [\e[3;4mWIDTH\e[0m] [\e[3;4mHEIGHT\e[0m] "
					"[\e[3;4mSEED\e[0m] [\e[3;4mFILE_OUT\e[0m]\n"
		"  \e[1mterrain_gen --stairs\e[0m [\e[3;4mWIDTH\e[0m] [\e[3;4mHEIGHT\e[0m] "
					"[\e[3;4mFILE_OUT\e[0m]\n"
		"  \e[1mterrain_gen --run\e[0m [\e[3;4mFILE_IN\e[0m]\n"
	);
}