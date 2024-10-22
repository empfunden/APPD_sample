# CMakeList.txt : CMake project for cmake_HH_example, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

project ("cmake_HH_example" VERSION 0.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(Boost REQUIRED program_options)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(TBB)
add_definitions( -DBOOST_ALL_DYN_LINK )
find_package (Eigen3 3.3 REQUIRED)
message("tbb lib: ${TBB_LIBRARIES}")
message("boost lib: ${Boost_LIBRARIES}")
message("boost inc:${Boost_INCLUDE_DIR}")
# find_package(Matlab REQUIRED ENG_LIBRARY MAT_LIBRARY)
find_package(nlohmann_json 3.2.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIR} ${Matlab_INCLUDE_DIRS} ${TBB_INCLUDE_DIR})
# Add source to this project's executable.
#configure_file(particle_method.h.in particle_method.h)

set(CMAKE_CXX_FLAGS "-Wno-narrowing")

add_executable(cmake_HH_example minimal_HH_neuron_example.cpp particle_io.cpp particle_method.cpp particle_method.h set_hh_eqn.h)
# TODO: Add tests and install targets if needed.

target_include_directories(cmake_HH_example PUBLIC
                           "${PROJECT_BINARY_DIR}"
)
target_link_libraries (cmake_HH_example 
Eigen3::Eigen
Boost::program_options
# ${Matlab_MAT_LIBRARY} ${Matlab_MX_LIBRARY} ${Matlab_ENG_LIBRARY}
nlohmann_json::nlohmann_json
TBB::tbb
)