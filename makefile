CC = nvcc
CFLAGS = -I./src -lm --gpu-architecture=sm_80 -m64

MODULES := $(wildcard src/*.cu)

bin/main: main.cu $(MODULES)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -rf bin

.PHONY: run
run: bin/main
	./$^
