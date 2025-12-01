CC          = g++
CFLAGS      = -g -fopenmp -march=skylake-avx512 -Iinclude
LIBDIR      = include
LDFLAGS     = -L$(LIBDIR) -lgemm -lm -fopenmp

SRCS        = $(wildcard *.cpp)
OBJS        = $(SRCS:.cpp=.o)
TARGETS     = $(SRCS:.cpp=.out)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.out: %.o
	$(CC) $< -o $@ $(LDFLAGS)

all: $(TARGETS)

clean:
	rm -rf *.o *.out
