# **Graduation Thesis: GIAdog**

## **Pre-requisitos**

 * `g++ >=8.3.0`
 * `python >=3.7.0`

## **GENERACION DE TERRENOS**

La generacion de terrenos en una simulacion permite al agente GIAdog aprender a moverse
antes de pasar al robot real. Los terrenos son almacenados como archivos `.txt` que 
contienen una matriz tal que cada posicion indica la altura del terreno en dicho pixel.

### **Compilacion**

Desde el directorio raiz del repositorio ejecute

```bash
make terrain_gen
```

Esto generara un archivo `terrain_gen` en el directorio `bin/`. 

### **Ejecucion**

La sintaxis del archivo binario `terrain_gen` es

```bash
terrain_gen ACTION ARGUMENT ...
```

Donde `ACTION` puede ser

 * `--hills` genera un terreno de colinas irregulares creados usando el Ruido de 
 Perlin. Sus argumentos son:

	* `ROUGHNESS` es un numero flotante que indica la aspereza del terreno. Su valor 
	se encuentra en el rango [0, 1], siendo 0 liso y 1 muy aspero.
	* `FREQUENCY` es un numero flotante que indica la frecuencia en la aparicion de 
	colinas. Su valor se encuentra en el rango [0, 1], siendo 0 que no hay colinas y 
	1 que hay muchas colinas juntas.
	* `HEIGHT` es un numero flotante no negativo que indica la altura maxima de las 
	colinas
	* `SEED` es un numero entero no negativo que indica la semilla para la 
	aleatoriedad al aplicar el Ruido de Perlin.
	* `FILE_OUT` es el archivo de texto donde se almacenara el terreno resultante.

 * `--maincra` genera un terreno de cubos distribuidos siguiendo el Ruido de Perlin. 
 Sus argumentos son:

	* `WIDTH` es un numero entero positivo que indica el ancho de los cubos.
	* `HEIGHT` es un numero flotante no negativo que indica la altura maxima de los
	cubos.
	* `SEED` es un numero entero no negativo que indica la semilla para la 
	aleatoriedad al aplicar el Ruido de Perlin.
	* `FILE_OUT` es el archivo de texto donde se almacenara el terreno resultante.

 * `--stairs` genera un terreno de escaleras. Sus argumentos son:

	* `WIDTH` es un numero entero positivo que indica el ancho de los escalones.
	* `HEIGHT` es un numero flotante positivo que indica la altura de los escalones.
	* `FILE_OUT` es el archivo de texto donde se almacenara el terreno resultante.

 * `--run` ejecuta el interpretador para crear un terreno personalizado. Las 
 instrucciones que lee el interpretador se explicaran mas adelante. Sus argumentos
 son:

	* `FILE_IN` es el archivo que contiene las instrucciones que leera el 
	interpretador

### **Interpretador**

Permite crear terrenos personalizados. Las instrucciones que reconoce son:

 * `TERRAIN ROWS COLS` inicializa el terreno con `ROWS` filas y `COLS` columnas,
 ambos numeros enteros positivos.

 * `PERLIN HEIGHT SMOOTH FREQUENCY SEED` aplica sobre el terreno el Ruido de Perlin
 tal que `HEIGHT`, `SMOOTH` y `FREQUENCY` son flotantes no negativos y `SEED` es un
 entero no negativo.

 * `STEP ROW COL WIDTH LENGTH HEIGHT` agrega al terreno un bloque en la posicion 
 (`ROW`, `COL`) siendo ambos enteros no negativos, al igual que `WIDTH` y `LENGTH`;
 y `HEIGHT` un flotante no negativo. 

 * `HILL ROW COL RADIUM HEIGHT CURVATURE` agrega al terreno una colina
 redonda cuyo centro se encuentra en la posicion (`ROW`, `COL`) siendo ambos enteros 
 no negativos, y `RADIUM`, `HEIGHT` y `CURVATURE` flotantes no negativos.

 * `STAIR ROW COL ORIENTATION WIDTH LENGTH HEIGH N` crea una escalera desde la 
 posicion (`ROW`, `COL`) con orientacion `ORIENTATION` el cual puede ser `N`, `S`, 
 `E` o `W` para indicar norte, sur, este u oeste respectivamente. `WIDTH` y `LENGHT`
 son enteros positivos que indican el ancho (numero de filas que ocupa) y largo 
 (numero de columnas que ocupa). `HEIGHT` es un flotante positivo que indica la 
 altura de cada escalon y `N` es un entero no negativo que indica el numero de 
 escalones.

 * `SAVE FILENAME` guarda el terreno actual en el archivo `FILENAME`.

Algunos ejemplos de terrenos personalizados es almacenan en el directorio 
`terrains/`.

### **Pruebas**

Para visualizar los terrenos generados se puede ejecutar

```bash
python src/simulation.py --test TERRAIN GIADOG ROW COL
```

Donde

 * `TERRAIN` es el archivo que almacena el terreno.

 * `GIADOG` es el archivo que contiene la configuracion del agente.

 * (`ROW`, `COL`) son las coordenadas donde aparecera el agente.

### **Ejemplos**

```bash
bin/terrain_gen --hills 0.7 0.8 2.3 42 a.txt
```

![Hills example](docs/terrain_examples/hills_example.png) 


```bash
bin/terrain_gen --maincra 15 1.2 1999 a.txt
```

![Maincra example](docs/terrain_examples/maincra_example.png)

```bash
bin/terrain_gen --stairs 11 0.07 a.txt
```

![Stairs example](docs/terrain_examples/stairs_example.png)

```bash
bin/terrain_gen --run terrains/tower.tg
```

![Tower](docs/terrain_examples/tower.png)

## **AUTORES**

 * Amin Arriaga
 * Eduardo Lopez