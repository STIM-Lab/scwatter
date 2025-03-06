#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 3.16)

#Name your project here
project(sCWatter LANGUAGES CUDA CXX)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

# fix warnings with BOOST and CMake
set(Boost_NO_WARN_NEW_VERSIONS 1)
find_package(Boost REQUIRED COMPONENTS program_options)


#find the TIRA library
set(CMAKE_CUDA_ARCHITECTURES 75)
find_package(CUDA REQUIRED)


find_package(glm CONFIG REQUIRED)
find_package(TIRA REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED core glfw-binding opengl3-binding)
find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(MKL CONFIG REQUIRED)
find_package(PNG)
find_package(JPEG)

#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif ( MSVC )

#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${TIRA_INCLUDE_DIRS}
			${CUDA_INCLUDE_DIRS}
			${Boost_INCLUDE_DIRS}
			${MKL_H}
)
link_directories(${CUDA_BIN_DIRS})
				
#create an executable
add_executable(scwatterplane
					CoupledWaveStructure.h
					sCWatterPlane.cpp
					cnpy/cnpy.cpp)

add_executable(scwatterlayer
					CoupledWaveStructure.h
					sCWatterLayer.cpp)

add_executable(scwatter
					CoupledWaveStructure.h
					FourierWave.h
					sCWatter.cpp
					third_Lapack.h
					cnpy/cnpy.cpp)

add_executable(scwatterview
					CoupledWaveStructure.h
					FourierWave.h
					third_Lapack.h
					cpuEvaluator.cpp
					cpuEvaluator.h
					gpuEvaluator.cu
					gpuEvaluator.h
					sCWatterView.cpp
					ImGuiFileDialog/ImGuiFileDialog.cpp
					cnpy/cnpy.cpp)

target_link_libraries(scwatterplane
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
)


target_link_libraries(scwatterlayer
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
			Eigen3::Eigen
)


target_link_libraries(scwatter
			ZLIB::ZLIB
			glm::glm
			Boost::program_options
			Eigen3::Eigen
			FFTW3::fftw3
			MKL::MKL
)

target_link_libraries(scwatterview
			ZLIB::ZLIB
			glfw
			glm::glm
			GLEW::GLEW
			${OPENGL_LIBRARIES}
			imgui::imgui
			Boost::program_options
			Eigen3::Eigen
			FFTW3::fftw3
			MKL::MKL
			PNG::PNG
			JPEG::JPEG
)