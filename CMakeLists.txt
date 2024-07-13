cmake_minimum_required(VERSION 3.4.0)

project(pisathon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64 -mssse3 -madx")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=generic")
#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")

# Speed up the build by removing CLI tools, tests, and benchmarks
set(PISA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(PISA_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(PISA_ENABLE_BENCHMARKING OFF CACHE BOOL "" FORCE)

find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIRS})

add_subdirectory(pisa)

add_library(_pisathon MODULE src/pyterrier_pisa/_pisathon.cpp)
target_link_libraries(_pisathon pisa)
#set_target_properties(_pisathon PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
#                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

#set_property(GLOBAL APPEND PROPERTY PY_LINKED_MODULES_LIST tbb)

python_extension_module(_pisathon)

#python_modules_header(modules
#                      FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
#include_directories(${modules_INCLUDE_DIRS})
#target_link_libraries(_pisathon ${linked_module_list} ${Boost_LIBRARIES})

install(TARGETS _pisathon LIBRARY DESTINATION pyterrier_pisa)
