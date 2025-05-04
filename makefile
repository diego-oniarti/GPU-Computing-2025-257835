CC = nvcc
CFLAGS = -g -I./src -lm --gpu-architecture=sm_80 -m64

MODULES := $(wildcard src/*.cu)
HEADERS := $(wildcard src/*.h)

bin/main: main.cu $(MODULES) $(HEADERS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $< $(MODULES)

.PHONY: clean
clean:
	rm -rf bin

.PHONY: run
run: bin/main
	./$^ $(FILE)
