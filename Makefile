SRCS := bpe.c txt2img.m
LIBS := -framework CoreML -framework Foundation -fsanitize=address

txt2img: $(SRCS)
	cc -Wall -Wextra -g -o $@ $^ $(LIBS)
