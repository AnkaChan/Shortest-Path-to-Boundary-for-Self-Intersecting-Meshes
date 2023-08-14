cmake_minimum_required (VERSION 3.0)
project (Shortest-Path-Test)


find_package(Eigen3 REQUIRED)
find_package(MeshFrame2 REQUIRED PATHS "./3rdParty/MeshFrame2" )
find_package(embree 3.0 REQUIRED)

set(THIRD_PARTY_INCLUDE_DIRS
        ${EIGEN3_INCLUDE_DIR}
		${MESHFRAME_INCLUDE_DIR} 
		${EMBREE_INCLUDE_DIRS}
		"./3rdParty/oneTBB/include"
		"./3rdParty/CuMatrix"
        )
		
include_directories(${THIRD_PARTY_INCLUDE_DIRS})
		

file(GLOB SHORTEST_PATH_SRCS
	"./ShortestPath/TetMesh/**.h"
	"./ShortestPath/TetMesh/**.cpp"
	"./ShortestPath/CollisionDetector/*.h"
	"./ShortestPath/CollisionDetector/*.cpp"
	"./ShortestPath/common/math/constants.cpp"
	${MESHFRAME_SOURCE_CPP_UTILITY}
)

set (SHORTEST_PATH_LIBS
	${EMBREE_LIBRARY}
	${embree_DIR}/../../tbb12.lib
	)

file(GLOB SRC
    "*.h"
    "*.cpp"
	"*.c"
)

add_executable(Shortest-Path-Test 
	${SRC}
	${SHORTEST_PATH_SRCS}
)
target_link_libraries(Shortest-Path-Test
	${SHORTEST_PATH_LIBS}
)