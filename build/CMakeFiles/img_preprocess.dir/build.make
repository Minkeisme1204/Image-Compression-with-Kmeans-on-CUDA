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
include CMakeFiles/img_preprocess.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/img_preprocess.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/img_preprocess.dir/flags.make

CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.o: CMakeFiles/img_preprocess.dir/flags.make
CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.o: ../lib/img_preprocess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.o -c /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/lib/img_preprocess.cpp

CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/lib/img_preprocess.cpp > CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.i

CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/lib/img_preprocess.cpp -o CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.s

# Object files for target img_preprocess
img_preprocess_OBJECTS = \
"CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.o"

# External object files for target img_preprocess
img_preprocess_EXTERNAL_OBJECTS =

libimg_preprocess.a: CMakeFiles/img_preprocess.dir/lib/img_preprocess.cpp.o
libimg_preprocess.a: CMakeFiles/img_preprocess.dir/build.make
libimg_preprocess.a: CMakeFiles/img_preprocess.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libimg_preprocess.a"
	$(CMAKE_COMMAND) -P CMakeFiles/img_preprocess.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/img_preprocess.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/img_preprocess.dir/build: libimg_preprocess.a

.PHONY : CMakeFiles/img_preprocess.dir/build

CMakeFiles/img_preprocess.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/img_preprocess.dir/cmake_clean.cmake
.PHONY : CMakeFiles/img_preprocess.dir/clean

CMakeFiles/img_preprocess.dir/depend:
	cd /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build /home/minkescanor/Desktop/WORKPLACE/Hust/Image-Kmeans-Compress/build/CMakeFiles/img_preprocess.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/img_preprocess.dir/depend

