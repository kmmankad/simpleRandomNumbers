# simpleRandomNumbers
# /*
#  * Name: simpleRandomNumbers
#  * File: Makefile
#  * Description: A minimalist Makefile to build this project
#  * Author: kmmankad (kmmankad@gmail.com kmankad@ncsu.edu)
#  * License: MIT License
#  */

TARGET ?= simpleRandomNumbers
SM_ARCH ?= sm_30

CC := nvcc
CCFLAGS := -DDEBUG=1 -arch=$(SM_ARCH)

all: clean $(TARGET)

$(TARGET): $(wildcard *.cu)
	$(CC) $(CCFLAGS) -o $@ $^

clean:
	rm -rf $(TARGET)
