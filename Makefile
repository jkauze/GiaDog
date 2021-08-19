CC=g++

.PHONY: clean

terrain_gen: src/terrain_gen.cpp
	$(CC) -std=c++17 $^ -o bin/$@.out -lm -lstdc++

clean:
	rm -rf bin/terrain_gen
