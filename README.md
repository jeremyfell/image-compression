Image Compression
=================
An implementation of a lossy image compression algorithm similar to JPEG, using Huffman coding for underlying lossless compression. Uses the NumPy library, and the Tkinter GUI library for demonstration purposes. Supports 24-bit uncompressed bitmap images, which are compressed to a custom .IMG format.

Usage
-----
To use, run `python3 image-compression.py`. Select either a .bmp or a .IMG file from the open file prompt. If the file is .bmp, the program will first ask you to input a quality factor, which is an integer between 1 and 100 (100 is best quality). The program will read the file, compress it, save the compressed file as a .IMG file, uncompress the file, and then display the original and reconstructed images side by side. It will also display the compression ratio (original file size / compressed file size) and the Peak Signal to Noise Ratio in dB. If the file is .IMG, the program will read the file, uncompress it, and then display the image.

Implementation
--------------
After reading in the data from a 24-bit uncompressed bitmap file, the algorithm works as follows:

![diagram](https://user-images.githubusercontent.com/31748813/100970099-1b997c00-34e9-11eb-8ee3-bb72ab35523f.png)

First, the RGB image data is converted into YUV color. The U and V chrominance channels are then subsampled, using 4:2:0 subsampling: only the top left pixel of each 2x2 block is kept, to reduce each channel to one quarter its previous size. Next, each of the three channels have padding added to their bottom and right edges, so that both their height and width are evenly divisible by 8. The 2D Discrete Cosine Transform is then applied to each 8x8 block of all three channels.

Afterwards, quantization is performed on the resulting component values. This starts with a base quantization matrix, which is from the IJG (Independent JPEG Group).

![matrix](https://user-images.githubusercontent.com/31748813/100970661-33253480-34ea-11eb-9f60-d1521535bf1c.png)

Let Q be the quality value provided to the algorithm. The base quantization matrix above is multiplied by a scalar S, which is determined by the following formula used by JPEG:

<img src="https://user-images.githubusercontent.com/31748813/100971242-543a5500-34eb-11eb-9534-89994694fdbc.png" height="50px"/>

Every entry of the matrix is also rounded down after this multiplication. Each 8x8 block of each channel is then multiplied by the computed quantization matrix, and all intensity values are rounded to be integers. Next, runlength encoding is applied to each 8x8 block of every channel, using a zigzag scan, to obtain a sequence of (skip, value) pairs. The runlength encoding values for the two chrominance channels are then appended together. Huffman encoding is applied separately to the luminance channel, and to the combined chrominance channels. The Huffman trees and encoded values are then saved to a file using the format specified in the next section.

Decompression uses the same pipeline but in reverse, and performing the inverse of each step. The main distinction between this algorithm and JPEG is the use of DPCM (Differential Pulse-Code Modulation) on the DC components obtained after DCT in JPEG compression.

File format
-----------
The file format used to store the compressed image is simple, with little overhead. It consists of a short 20-byte file header, followed by two blocks of Huffman tree data, followed by two blocks of encoded image data.

The file header is specified as follows:
|     Starting byte number    |     Name of stored value                         |     Size in bytes    |     Description                                                                                                            |
|-----------------------------|--------------------------------------------------|----------------------|----------------------------------------------------------------------------------------------------------------------------|
|     0                       |     File   signature                             |     3                |     This value is always set to the   ASCII string ‘IMG’. It is simply used to identify that the file is an IMG   file.    |
|     3                       |     Quality   factor                             |     1                |     An integer between 1 and 100                                                                                           |
|     4                       |     Height                                       |     2                |     The height of the image in   pixels                                                                                    |
|     6                       |     Width                                        |     2                |     The width of the image in pixels                                                                                       |
|     8                       |     Luminance   Huffman tree size                |     2                |     The size in bytes of the Huffman   tree for the luminance channel                                                      |
|     10                      |     Chrominance   Huffman tree size              |     2                |     The size in bytes of the Huffman   tree for the chrominance channels                                                   |
|     12                      |     Luminance   Huffman-encoded values size      |     4                |     The size in bytes of the block   of encoded values for the luminance channel                                           |
|     16                      |     Chrominance   Huffman-encoded values size    |     4                |     The size in bytes of the block   of encoded values for the chrominance channels                                        |

The remaining file is stored as follows:
| Section                                     |
|---------------------------------------------|
|     Luminance Huffman tree                  |
|     Chrominance   Huffman tree              |
|     Luminance   Huffman-encoded values      |
|     Chrominance   Huffman-encoded values    |

Since each of these four blocks may contain a number of bits that is not evenly divisible into bytes, it is necessary to add padding bits when the bits cannot be evenly divided by 8. A simple padding method fixes this issue. Suppose that N is the size of the block in bits, and let M = N % 8. Then 8 – M padding bits are added to the start of the block. All of these are `0` bits, except that the last is a `1` bit. When a new section is read, every `0` bit is discarded until a `1` bit is encountered (which is also discarded). All data after that `1` bit is the actual data.

Each Huffman tree is stored as a bitstring that describes a depth-first traversal of the tree. The traversal starts at the root of the tree, and uses a recursive algorithm. For each node, if it is an internal node, a `0` bit is stored. The bitstring for a traversal of the left child node is then stored, followed by the bitstring for a traversal of the right child node. If the node is a leaf node, a `1` bit is stored, followed by a 12-bit signed integer. 12 bits are used because the maximum value of a component value after DCT in an 8x8 block is 2040; component values can be negative as well, so 12 bits can store integers in the range [-2048, 2047] which is sufficient.
