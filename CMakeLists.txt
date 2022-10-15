cmake_minimum_required(VERSION 3.9)
project(diffusionapp VERSION 0.1.0)

# Set bundle properties
set(MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION})
set(MACOSX_BUNDLE_INFO_STRING "Native mac interface for stable diffusion")
set(MACOSX_BUNDLE_GUI_IDENTIFIER "org.cqcumbers.diffusionapp")

# Link diffusionapp against dependencies
add_executable(diffusionapp MACOSX_BUNDLE main.m txt2img.m bpe.c)
target_compile_options(diffusionapp PRIVATE -g "-fsanitize=address")
target_link_libraries(diffusionapp PRIVATE "-framework Cocoa"
    "-framework CoreML" "-framework Foundation" "-fsanitize=address")
