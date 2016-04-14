# CMakeLists.txt
# Date: 2014-08-06
# Author: Karsten Ahnert (karsten.ahnert@gmx.de)
#

project ( bdf2 )
cmake_minimum_required ( VERSION 2.8 )

find_package ( Boost REQUIRED )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )        ## Optimize

include_directories ( ${Boost_INCLUDE_DIRS} )

add_executable ( bdf2 bdf2.cpp )