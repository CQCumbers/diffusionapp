cmake_minimum_required(VERSION 3.19)
project(diffusionapp VERSION 0.1.0)
include(FetchContent)

# Set bundle properties
set(MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION})
set(MACOSX_BUNDLE_INFO_STRING "Native mac interface for stable diffusion")
set(MACOSX_BUNDLE_GUI_IDENTIFIER "org.cqcumbers.diffusionapp")

# Download models at configure time
set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(models
    URL https://f002.backblazeb2.com/file/cqcumbers-public-b2/diffusionapp_models.zip
    URL_HASH SHA1=b48f48e451d2e90b0fce0a27ae656c92d5990a5e
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")
FetchContent_MakeAvailable(models)
file(GLOB_RECURSE RESOURCES "models/*")

# Link diffusionapp against dependencies
add_executable(diffusionapp MACOSX_BUNDLE main.m txt2img.m bpe.c ${RESOURCES})
target_compile_options(diffusionapp PRIVATE -g "-fsanitize=address")
target_link_libraries(diffusionapp PRIVATE "-framework Cocoa"
    "-framework CoreML" "-framework Foundation"
    "-framework UniformTypeIdentifiers" "-fsanitize=address")

# Set resource paths
foreach (FILE ${RESOURCES})
    file(RELATIVE_PATH NEW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models" ${FILE})
    get_filename_component(NEW_DIR ${NEW_PATH} DIRECTORY)
    set_property(SOURCE ${FILE} PROPERTY MACOSX_PACKAGE_LOCATION "Resources/${NEW_DIR}")
    source_group(Resources/${NEW_DIR} FILES ${FILE})
endforeach()
