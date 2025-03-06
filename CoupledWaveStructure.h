#pragma once

#include "tira/optics/planewave.h"
#include <vector>
#include <fstream>

/// Coupled Wave structure file format
// [64-bit unsigned int]	precision (number of bytes, 4 = float, 8 - double)
// [64-bit unsigned int]    number of incident plane waves
// FOR EACH INCIDENT PLANE WAVE
//		[N-bit float]			E0.x (real)
//		[N-bit float]			E0.x (imaginary)
//		[N-bit float]			E0.y (real)
//		[N-bit float]			E0.y (imaginary)
//		[N-bit float]			k.x (real)
//		[N-bit float]			k.x (imaginary)
//		[N-bit float]			k.y (real)
//		[N-bit float]			k.y (imaginary)
//		[N-bit float]			k.z (real)
//		[N-bit float]			k.z (imaginary)

//		[64-bit unsigned int]	number of homogeneous layers

// FOR EACH LAYER
//		[N-bit float]			z coordinate
//		[64-bit unsigned int]	number of reflected plane waves
//		FOR EACH REFLECTED PLANE WAVE
//			[N-bit float]			E0.x (real)
//			[N-bit float]			E0.x (imaginary)
//			[N-bit float]			E0.y (real)
//			[N-bit float]			E0.y (imaginary)
//			[N-bit float]			k.x (real)
//			[N-bit float]			k.x (imaginary)
//			[N-bit float]			k.y (real)
//			[N-bit float]			k.y (imaginary)
//			[N-bit float]			k.z (real)
//			[N-bit float]			k.z (imaginary)	
//		[64-bit unsigned int]	number of transmitted plane waves
//		FOR EACH TRANSMITTED PLANE WAVE
//			[N-bit float]			E0.x (real)
//			[N-bit float]			E0.x (imaginary)
//			[N-bit float]			E0.y (real)
//			[N-bit float]			E0.y (imaginary)
//			[N-bit float]			k.x (real)
//			[N-bit float]			k.x (imaginary)
//			[N-bit float]			k.y (real)
//			[N-bit float]			k.y (imaginary)
//			[N-bit float]			k.z (real)
//			[N-bit float]			k.z (imaginary)	

template <typename T>
struct HomogeneousLayer {
	T z;										// z coordinate of the layer
	std::vector < tira::planewave<T> > Pr;		// plane waves reflected off of the boundary (along -z)
	std::vector < tira::planewave<T> > Pt;		// plane wave transmitted through the boundary (along z)
};

template <typename T>
class CoupledWaveStructure {
public:

	std::vector< tira::planewave<T> > Pi;		// incoming plane waves

	std::vector< HomogeneousLayer<T> > Layers;			// homogeneous sample layers and their respective plane waves

	void save(std::string filename) {
		std::ofstream file(filename, std::ios::out | std::ios::binary);

		size_t sizeof_T = sizeof(T);
		file.write((char*)&sizeof_T, sizeof(size_t));		// output the precision (float = 4, double = 8)

		size_t sizeof_Pi = Pi.size();
		file.write((char*)&sizeof_Pi, sizeof(size_t));		// output the number of incident plane waves
		for (size_t iPi = 0; iPi < Pi.size(); iPi++) {						// for each plane wave
			file.write((char*)&Pi[iPi], sizeof(tira::planewave<T>));		// output the plane wave
		}
		size_t sizeof_Layers = Layers.size();
		size_t sizeof_Pr, sizeof_Pt;						// Pr/Pt has a dim of M. While the inside field has a dim of M*2M
		file.write((char*)&sizeof_Layers, sizeof(size_t));	// output the number of homogeneous layers
		for (size_t iLayers = 0; iLayers < sizeof_Layers; iLayers++) {
			file.write((char*)&Layers[iLayers].z, sizeof(T));		// output the layer position
			sizeof_Pr = Layers[iLayers].Pr.size();
			file.write((char*)&sizeof_Pr, sizeof(size_t));		// output the number of reflected plane waves
			for (size_t iPr = 0; iPr < Layers[iLayers].Pr.size(); iPr++) {
				file.write((char*)&Layers[iLayers].Pr[iPr], sizeof(tira::planewave<T>));
			}

			sizeof_Pt = Layers[iLayers].Pt.size();
			file.write((char*)&sizeof_Pt, sizeof(size_t));		// output the number of transmitted plane waves
			for (size_t iPt = 0; iPt < Layers[iLayers].Pt.size(); iPt++) {
				file.write((char*)&Layers[iLayers].Pt[iPt], sizeof(tira::planewave<T>));
			}
		}
		file.close();
	}

	bool load(std::string filename) {
		std::ifstream file(filename, std::ios::in | std::ios::binary);
		if (!file) return false;

		size_t sizeof_T;
		file.read((char*)&sizeof_T, sizeof(size_t));						// read the precision (float = 4, double = 8)
		size_t sizeof_Pi;
		file.read((char*)&sizeof_Pi, sizeof(size_t));						// read the number of incident plane waves
		Pi.resize(sizeof_Pi);												// allocate space for the incident plane waves
		for (size_t iPi = 0; iPi < Pi.size(); iPi++) {
			file.read((char*)&Pi[iPi], sizeof(tira::planewave<T>));
		}
		size_t sizeof_Layers;
		size_t sizeof_Pr, sizeof_Pt;
		file.read((char*)&sizeof_Layers, sizeof(size_t));					// read the number of homogeneous layers
		Layers.resize(sizeof_Layers);										// allocate space for the layer structures
		for (size_t iLayers = 0; iLayers < sizeof_Layers; iLayers++) {
			file.read((char*)&Layers[iLayers].z, sizeof(T));				// read the layer position
			file.read((char*)&sizeof_Pr, sizeof(size_t));					// read the number of reflected plane waves
			Layers[iLayers].Pr.resize(sizeof_Pr);							// allocate space for the reflected waves
			for (size_t iPr = 0; iPr < Layers[iLayers].Pr.size(); iPr++) {
				file.read((char*)&Layers[iLayers].Pr[iPr], sizeof(tira::planewave<T>));
			}
			file.read((char*)&sizeof_Pt, sizeof(size_t));					// read the number of transmitted plane waves
			Layers[iLayers].Pt.resize(sizeof_Pt);							// allocate space for the transmitted plane waves
			for (size_t iPt = 0; iPt < Layers[iLayers].Pt.size(); iPt++) {
				file.read((char*)&Layers[iLayers].Pt[iPt], sizeof(tira::planewave<T>));
			}
		}
		file.close();
		return true;
	}

	/// <summary>
	/// Get the plane index from the given z coordinate
	/// </summary>
	/// <param name="z">spatial z coordinate to be tested</param>
	/// <returns>Returns an index to the next plane along the positive z axis </returns>
	size_t getPlaneIndex(T z) {
		for (size_t l = 0; l < Layers.size(); l++) {
			if (z >= Layers[l].z)
				return l + 1;
		}
		return 0;
	}
};