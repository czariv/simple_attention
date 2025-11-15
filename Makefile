CC          = g++
CFLAGS      = -g -fopenmp -mavx512f -Ilib/include
LIBDIR      = lib/lib
LDFLAGS     = -L$(LIBDIR) -lopenblas -lm -fopenmp

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
