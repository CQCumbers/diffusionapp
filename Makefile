SRCS := bpe.c txt2img.m

txt2img: $(SRCS)
	cc -Wall -Wextra -o $@ $^
