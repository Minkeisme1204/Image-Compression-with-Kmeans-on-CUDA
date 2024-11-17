# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build

# Include any dependencies generated for this target.
include CMakeFiles/kmeans_gpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kmeans_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kmeans_gpu.dir/flags.make

CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.o: CMakeFiles/kmeans_gpu.dir/flags.make
CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.o: ../lib/kmeans_gpu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.o"
	/usr/local/cuda-12.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/lib/kmeans_gpu.cu -o CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.o

CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target kmeans_gpu
kmeans_gpu_OBJECTS = \
"CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.o"

# External object files for target kmeans_gpu
kmeans_gpu_EXTERNAL_OBJECTS =

libkmeans_gpu.a: CMakeFiles/kmeans_gpu.dir/lib/kmeans_gpu.cu.o
libkmeans_gpu.a: CMakeFiles/kmeans_gpu.dir/build.make
libkmeans_gpu.a: CMakeFiles/kmeans_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libkmeans_gpu.a"
	$(CMAKE_COMMAND) -P CMakeFiles/kmeans_gpu.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kmeans_gpu.dir/build: libkmeans_gpu.a

.PHONY : CMakeFiles/kmeans_gpu.dir/build

CMakeFiles/kmeans_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kmeans_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kmeans_gpu.dir/clean

CMakeFiles/kmeans_gpu.dir/depend:
	cd /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build/CMakeFiles/kmeans_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kmeans_gpu.dir/depend

