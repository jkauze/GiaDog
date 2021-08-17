/*
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 2021/08/13

	Terrain generator.
*/
#pragma once 

// Utilities
#include <map>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include "utils.hpp"

// Terrain
#include "perlin_noise_t.hpp"

#define ADAPT_TERRAIN_SIZE 500
#define MAX_ROUGHNESS 0.3
#define MAX_FREQUENCY 4.5

class terrain_t {
	private:
		// Atributos del terreno.
		unsigned rows_, cols_;
		std::vector<std::vector<float>> terrain_;


	public:
		/* Inicializa el terreno vacio. */
		terrain_t(void);
		/* Inicializa el terreno vacio y lee las instrucciones de un archivo. */
		terrain_t(std::string filename);

		/*
			Genera el terreno inicial.

			Args:
				rows: unsigned  ->  Numero de filas del mapa.
				cols: unsigned  ->  Numero de columnas del mapa.
		*/
		void terrain(unsigned rows, unsigned cols);
		void terrain(std::vector<std::string> args);

		/*
			Agrega el Perlin Noise al terreno.

			Args:
				h: float  ->  Maxima altura de las colinas.
				smooth: float  ->  Suavidad del terreno usando Perlin Noise. Debe ser 
					mayor a 0.
				f: float  ->  Frecuencia de las colinas.
				seed: uint32_t ->  Semilla para la aleatoriedad del Perlin Noise.
		*/
		void perlin(float h, float smooth, float f, uint32_t seed);
		void perlin(std::vector<std::string> args);

		/*
			Agrega un 'cubo' al terreno.

			Args:
				row: unsigned  ->  Fila donde se encuentra la esquina superior izquierda.
				col: unsigned  ->  Columna donde se encuentra la esquina superior 
					izquierda.
				w: unsigned  ->  Ancho del cubo.
				l: unsigned  ->  Largo del cubo.
				h: float  ->  Altura del cubo.
		*/
		void step(unsigned row, unsigned col, unsigned w, unsigned l, float h);
		void step(std::vector<std::string> args);

		/*
			Agrega una colina redonda al terreno

			Args:
				row: unsigned  ->  Fila donde se encuentra el centro.
				col: unsigned  ->  Columna donde se encuentra el centro.
				r: float  ->  Radio
				h: float  ->  Altura.
				c: float  ->  Curvatura. Si es 0, la colina parecera un cono. Mientras 
					tienda a infinito, la colina tendera a un cilindro.
		*/
		void hill(unsigned row, unsigned col, float r, float h, float c);
		void hill(std::vector<std::string> args);

		/*
			Agrega una escalera al terreno.

			Args:
				row: unsigned  ->  Fila donde se encuentra la parte superior izquierda de 
					la linea que inicia la escalera.
				col: unsigned  ->  Columna donde se encuentra la parte superior izquierda 
					de la linea que inicia la escalera.
				o: char  ->  Orientacion de la escalera. Puede ser N (Norte), E (Este), S 
					(Sur) o W (Oeste).
				w: unsigned  ->  Ancho de cada escalon.
				l: unsigned  ->  Largo de cada escalon.
				h: float  ->  Altura de cada escalon.
				n: unsigned  ->  Numero de escalones.
		*/
		void stair(
			unsigned row, 
			unsigned col, 
			char o, 
			unsigned w, 
			unsigned l, 
			float h,
			unsigned n
		);
		void stair(std::vector<std::string> args);

		/* Guardamos el terreno en un archivo txt. */
		void save(std::string filename);
		void save(std::vector<std::string> args);

		/* Ejecutamos el interpretador. */
		void read_instructions(std::string filename);

		/*
			Generacion de terrenos de colinas adaptativos.

			Args:
				r: float  ->  Indice de aspereza del terreno.
				f: float  ->  Indice de frecuencia con la que aparecen colinas.
				h: float  ->  Altura maxima del terreno.
				seed: uint32_t ->  Semilla.
		*/
		void adaptative_hills(float r, float f, float h, uint32_t seed);

		/*
			Generacion de terrenos maincra adaptativos.

			Args:
				w: unsigned  ->  Ancho de cada bloque.
				h: float  ->  Altura maxima.
				seed: uint32_t ->  Semilla.
		*/
		void adaptative_maincra(unsigned w, float h, uint32_t seed);

		/*
			Generacion de terrenos de escaleras adaptativos.

			Args:
				w: unsigned  ->  Ancho de cada escalon.
				h: float  ->  Alto de cada escalon.
		*/
		void adaptative_stairs(unsigned w, float h);
};

terrain_t::terrain_t(void) {
	this->terrain_ = {{0}};
}

terrain_t::terrain_t(std::string filename) {
	this->terrain_ = {{0}};
	this->read_instructions(filename);
}

void terrain_t::terrain(unsigned rows, unsigned cols) {
	this->rows_ = rows;
	this->cols_ = cols;
	this->terrain_ = {};

	// Generamos el terreno plano con las dimensiones indicadas.
	for (int i = 0; i < rows; i++) 
	{
		std::vector<float> row;
		for (int j = 0; j < cols; j++) 
		{
			row.push_back(0.0);
		}
		this->terrain_.push_back(row);
	}
}

void terrain_t::terrain(std::vector<std::string> args) {
	if (args.size() != 2) 
	{
		error("TERRAIN instruction expected 2 arguments.");
	}
	this->terrain(std::stoul(args[0]), std::stoul(args[1]));
}

void terrain_t::perlin(float h, float f, float smooth, uint32_t seed) {
	// Si la altura maxima o la rugisidad son 0, entonces no se aplica el ruido.
	if (h == 0 || smooth == 0)
	{
		return;
	}

	// Aplicamos el Perlin Noise sobre el terreno.
	siv::perlin_noise_t perlin(seed);
	for (int i = 0; i < this->rows_; i++)
	{
		for (int j = 0; j < this->cols_; j++)
		{
			this->terrain_[i][j] += h * perlin.accumulatedOctaveNoise2D_0_1(
				(double) i * smooth / this->rows_, 
				(double) j * smooth / this->cols_, 
				f
			);
		}
	}
}

void terrain_t::perlin(std::vector<std::string> args) {
	if (args.size() != 4) 
	{
		error("PERLIN instruction expected 4 arguments.");
	}
	this->perlin(
		std::stof(args[0]), 
		std::stof(args[1]), 
		std::stof(args[2]), 
		std::stoi(args[3])
	);
}

void terrain_t::step(unsigned row, unsigned col, unsigned w, unsigned l, float h) {
	// Verificamos que el bloque no se salga de los limites.
	unsigned end_row = row + w > this->rows_ ? this->rows_ : row + w;
	unsigned end_col = col + l > this->cols_ ? this->cols_ : col + l;

	for (int i = row; i < end_row; i++) 
	{
		for (int j = col; j < end_col; j++)
		{
			this->terrain_[i][j] += h;
		}
	}
}

void terrain_t::step(std::vector<std::string> args) {
	if (args.size() != 5) 
	{
		error("STEP instruction expected 5 arguments.");
	}
	this->step(
		std::stoul(args[0]), 
		std::stoul(args[1]), 
		std::stoul(args[2]), 
		std::stoul(args[3]), 
		std::stof(args[4])
	);
}

void terrain_t::hill(unsigned row, unsigned col, float r, float h, float c) {
	// Verificamos que la circunferencia no se salga de los limites del terreno.
	unsigned begin_row = row - r < 0 ? 0 : row - r;
	unsigned begin_col = col - r < 0 ? 0 : col - r;
	unsigned end_row = row + r + 1 > this->rows_ ? this->rows_ : row + r + 1;
	unsigned end_col = col + r + 1 > this->cols_ ? this->cols_ : col + r + 1;
	unsigned d;

	for (int i = begin_row; i < end_row; i++) 
	{
		for (int j = begin_col; j < end_col; j++)
		{
			d = pow(i - row, 2) + pow(j - col, 2);
			if (d <= pow(r, 2))
			{
				this->terrain_[i][j] += powf((1 - d / pow(r, 2)) * h, 1 / (c + 1));
			}
		}
	}
}

void terrain_t::hill(std::vector<std::string> args) {
	if (args.size() != 5) 
	{
		error("HILL instruction expected 5 arguments.");
	}
	this->hill(
		(unsigned) std::stoul(args[0]), 
		(unsigned) std::stoul(args[1]), 
		std::stof(args[2]), 
		std::stof(args[3]), 
		std::stof(args[4])
	);
}

void terrain_t::stair(
	unsigned row, 
	unsigned col, 
	char o, 
	unsigned w, 
	unsigned l, 
	float h,
	unsigned n
) {
	switch(o)
	{
		case 'E':
			for (int i = 0; i < n; i++) 
			{
				this->step(row, col + i*l, w, l, i*h);
			}
			break;

		case 'S':
			for (int i = 0; i < n; i++) 
			{
				this->step(row + i*w, col, w, l, i*h);
			}
			break;

		case 'W':
			for (int i = 0; i < n; i++) 
			{
				this->step(row, col - (i+1)*l, w, l, i*h);
			}
			break;

		case 'N':
			for (int i = 0; i < n; i++) 
			{
				this->step(row - (i+1)*w, col + i*w, w, l, i*h);
			}
			break;

		default:
		  	error((std::string) "Undefined orientation \e[1m" + o + "\e[0m.");
	}
}

void terrain_t::stair(std::vector<std::string> args) {
	if (args.size() != 5) 
	{
		error("STAIR instruction expected 7 arguments.");
	}
	this->stair(
		std::stoul(args[0]), 
		std::stoul(args[1]), 
		args[2][0],
		std::stoul(args[3]), 
		std::stoul(args[4]), 
		std::stof(args[5]),
		std::stoul(args[6])
	);
}

void terrain_t::save(std::string filename) {
	std::ofstream file;
	file.open(filename);

	for (int i = 0; i < this->rows_; i++) 
	{
		for(int j = 0; j < this->cols_; j++) 
		{
			file << this->terrain_[i][j] << ", ";
		}
		file << "\n";
	}
	
	file.close();
}

void terrain_t::save(std::vector<std::string> args) {
	if (args.size() != 5) 
	{
		error("STAIR instruction expected 1 argument.");
	}
	this->save(args[0]);
}

void terrain_t::read_instructions(std::string filename) {
	std::string line, instr;
	std::vector<std::string> tokens, args;
	std::ifstream file(filename);

	// Leemos el archivo linea por linea,
	if (file.is_open())
	{
		while (getline(file, line))
		{
			tokens = split(line);

			// Ignoramos las lineas vacias o que comienzen con '#'.
			if (tokens.size() == 0 || tokens[0][0] == '#')
			{
				continue;
			}

			// Obtenemos la instruccion y sus argumentos.
			instr = tokens[0];
			args = {};
			for (int i = 1; i < tokens.size(); i++)
			{
				args.push_back(tokens[i]);
			}

			// Verificamos que la instruccion sea valida.
			if (instr == "TERRAIN") this->terrain(args);
			else if (instr == "PERLIN") this->perlin(args);
			else if (instr == "STEP") this->step(args);
			else if (instr == "HILL") this->hill(args);
			else if (instr == "STAIR") this->stair(args);
			else if (instr == "SAVE") this->save(args);
			else error("Undefined instruction \e[1m" + instr + "\e[0m.");
		}
		file.close();
	}
	else 
	{
		error("Unable to open file \e[1m" + filename + "\e[0m.");
	} 
}

void terrain_t::adaptative_hills(float r, float f, float h, uint32_t seed) {
	if (r < 0 || r > 1 || f < 0 || f > 1)
	{
		error("Roughness and frequency rate must be in [0,1].");
	}

	// Aplanamos el terreno.
	this->terrain(ADAPT_TERRAIN_SIZE, ADAPT_TERRAIN_SIZE);
	// Agregamos las colinas.
	this->perlin(h, 1, f * MAX_FREQUENCY, seed);
	// Agregamos la rugosidad.
	this->perlin(r * MAX_ROUGHNESS, 50, 15, seed+1);
}

void terrain_t::adaptative_maincra(unsigned w, float h, uint32_t seed) {

	this->terrain(ADAPT_TERRAIN_SIZE, ADAPT_TERRAIN_SIZE);
	siv::perlin_noise_t perlin(seed);

	unsigned row, col;
	float h_k;
	
	// Agregamos cada bloque siguiendo el ruido de Perlin.
	row = 0;
	while (row < this->rows_)
	{
		col = 0;
		while (col < this->cols_)
		{
			h_k = h * perlin.accumulatedOctaveNoise2D_0_1(
				(double) row * 15 / this->rows_, 
				(double) col * 15 / this->cols_, 
				1
			);
			this->step(row, col, w, w, h_k);
			col += w;
		}
		row += w;
	}
}

void terrain_t::adaptative_stairs(unsigned w, float h) {
	this->terrain(500, 500);

	// Ancho de la zona inicial
	unsigned init_w = 50;

	// Escaleras iniciales
	unsigned n0 = (100 - init_w - (100 - init_w) % w) / w;
	this->stair(0, 0, 'E', 500, w, h, n0);

	// Zona inicial
	this->step(0, n0 * w, 500, init_w + (100 - init_w) % w, n0 * h);

	// Resto de escaleras subiendo
	unsigned n1 = (200 - init_w) / w;
	this->step(0, 100, 500, 200 + 200 % w, (n0 + 1) * h);
	this->stair(0, 100, 'E', 500, w, h, n1);
	this->step(0, 100 + n1 * w, 500, init_w + (200 - init_w) % w + 200 % w, n1 * h);

	// Escaleras bajando
	unsigned n2 = 200 / w;
	this->step(0, 300 + 200 % w, 500, n2 * w, (n0 + 1) * h + n1 * h - n2 * h);
	this->stair(0, 500, 'W', 500, w, h, n2);
}