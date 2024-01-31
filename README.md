# scwatter

SCWatter aims to simulate the interactions between light and samples. The algorithm is based on coupled wave theory and wave propagation theory.

* sCWatter solvesthe field for 3D heterogeneous samples
* sCWatterplane solves the field for two homogeneous sample boundaries.
* sCWatterlayer solves the field for multiple homogeneous sample boundaries.
* sCWatterview visualize the simulation result from sCWatter/sCWatterplane/sCWatterlayer.

## Main Tools
* CUDA (https://developer.nvidia.com/cuda-toolkit)
* Intel oneAPIBase Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
* tiralib (https://github.com/STIM-Lab/tiralib)
* Vcpkg (https://vcpkg.io/en/getting-started.html)
	- You should be able to install all necessary libraries using vcpkg (https://vcpkg.io/en/packages)



## Workflow
1. Generate you sample. 
	* It should be a .npy file with the dimension of [Z, Y, X].
	* You can you tileSample.py to generate a 3D sample.

2. Select a proper executable from sCWatter, sCWatterplane, and sCWatterlayer. The calculated field will be saved as .cw file.

For example:
	
	```
	sCWatter --help 			# See parameter choices
	sCWatter --sample tilesample.npy --lambda 1 --coef 20 20 --size 64 64 20 --external
	```
	
3. Visualize the field using sCWatterview. For example:

	```
	sCWatterview --help  		# See parameter choices
	sCWatterview --input c.cw --size 100
	```
	Or a non-interactive version to save any plane
	```
	sCWatterview --input c.cw --size 100 --nogui --axis 2 --slice -10
	```

## Data Types
double/complex128.

## Notes
*The users can set all they need from the command line via the Boost library.*

*"executable - -help" will provide more information about the input, attributes, and output.*

