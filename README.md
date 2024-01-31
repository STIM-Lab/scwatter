# scwatter

SCWatter aims to simulate the interactions between light and samples. The algorithm is based on coupled wave theory and wave propagation theory.

* sCWatter solves the field for 3D heterogeneous samples
* sCWatterplane solves the field for two homogeneous sample boundaries.
* sCWatterlayer solves the field for multiple homogeneous sample boundaries.
* sCWatterview visualizes the simulation result from sCWatter/sCWatterplane/sCWatterlayer.

## Main Tools
* CUDA (https://developer.nvidia.com/cuda-toolkit)
* Intel oneAPIBase Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
* tiralib (https://github.com/STIM-Lab/tiralib)
* Vcpkg (https://vcpkg.io/en/getting-started.html)
	- You should be able to install all necessary libraries using vcpkg (https://vcpkg.io/en/packages)



## Workflow
1. Generate your sample. 
	* It should be a .npy file with the dimensions of [Z, Y, X].
	* You can you tileSample.py to generate a 3D sample.

2. Select a proper executable from sCWatter, sCWatterplane, and sCWatterlayer. The calculated field will be saved as .cw file. For example:

	```
	sCWatter --help 			# See parameter choices
	sCWatter --sample tilesample.npy --lambda 1 --coef 20 20 --size 64 64 20 --external --output sample.cw
	```
	
3. Visualize the field using sCWatterview interactively. For example:

	```
	sCWatterview --help  		# See parameter choices
	sCWatterview --input sample.cw --size 100
	```
	Or a non-interactive way to save any plane as a .npy file.
	```
	sCWatterview --input sample.cw --size 100 --nogui --axis 2 --slice -10 --output xy.npy
	```

4. (optional) If you select the "nogui" way, you can visualize the saved plane in Python. For example:

	```
	A = np.load("xy.npy")
	B = np.real(A[:, :, 1])
	plt.imshow(B, extent=[50, -50, -50, 50])
	plt.set_cmap("RdYlBu")
	plt.colorbar()
	plt.savefig("xy.png")
	```

