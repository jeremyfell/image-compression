# Import libraries
import sys
import os
import math
import heapq
import time
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
import tkinter.font
import numpy as np

################################
# Draw Image
################################

# Puts a single pixel into Tkinter image
def draw_pixel(canvas_image, pixel, x, y):
    canvas_image.put(hex(pixel), (x, y))

# Draws an RGB image to a Tkinter canvas image
def draw_image(width, height, image):
    # Create empty Tkinter image with correct dimensions
    canvas_image = tk.PhotoImage(width=width, height=height)

    # Draw each pixel
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            draw_pixel(canvas_image, pixel, x, y)

    return canvas_image

################################
# Color Conversion
################################

# Convert integer RGB values [0,255] to a hex color code string
def hex(pixel):
    (r, g, b) = pixel
    return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)

# Converts an image from RGB to YUV
def rgb_image_to_yuv(image):
    (height, width, _) = image.shape

    # RGB in range [0,255]
    pixels = image.reshape(height * width, 3)

    # RGB in range [0-1]
    pixels = pixels / 255
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    # Y in range [0,1]
    # U in range [-0.886,0.886]
    # V in range [-0.701,0.701]
    temp_r = 0.299*r
    temp_g = 0.587*g
    temp_b = 0.114*b
    y = temp_r + temp_g + temp_b
    u = -temp_r - temp_g + 0.886*b
    v = 0.701*r - temp_g - temp_b

    # YUV in range [0,255]
    y = y * 255
    u = ((u + 0.886 ) / 1.772) * 255
    v = ((v + 0.701) / 1.402) * 255

    yuv_pixels = np.stack([y, u, v], axis=1)

    yuv_image = yuv_pixels.reshape(height, width, 3)
    return np.round(yuv_image)

# Converts an image from YUV to RGB
def yuv_image_to_rgb(image):
    (height, width, _) = image.shape

    # YUV in range [0,255]
    pixels = image.reshape(height * width, 3)

    # YUV in range [0-1]
    pixels = pixels / 255
    y = pixels[:, 0]
    u = pixels[:, 1]
    v = pixels[:, 2]

    # Y in range [0,1]
    # U in range [-0.886,0.886]
    # V in range [-0.701,0.701]
    u = u * 1.772 - 0.886
    v = v * 1.402 - 0.701

    # RGB in range [0,255]
    r = (y + v) * 255
    g = (y - 0.19420*u - 0.50936*v) * 255
    b = (y + u) * 255

    rgb_pixels = np.stack([r, g, b], axis=1)
    rgb_pixels[rgb_pixels > 255] = 255
    rgb_pixels[rgb_pixels < 0] = 0

    rgb_image = rgb_pixels.reshape(height, width, 3)

    return np.round(rgb_image)

################################
# Padding
################################

# Pads the bottom and right edges of the channel so that the
# height and width are multiples of 8
def pad_edges(channel):
    (height, width) = channel.shape

    height_padding = math.ceil(height / 8) * 8 - height
    width_padding = math.ceil(width / 8) * 8 - width
    channel = np.pad(channel, pad_width=((0, height_padding), (0, width_padding)), mode='constant')

    return channel

# Removes the padding from the bottom and right edges of the channel
# to restore the original height and width
def unpad_edges(channel, height, width):
    return channel[0:height, 0:width]

################################
# Splitting and Recombining
################################

# Combines three channel arrays into one image array
def create_image_from_channels(r, g, b):
    image = np.stack([r, g, b], axis=2)
    return image

# Combine two channels into one channel with double the height
def combine_channels(channel1, channel2):
    return np.concatenate((channel1, channel2), axis=0)

# Split a channel (horizontally) into two channels
def split_channels(channels):
    (height, width) = channels.shape
    split = height // 2
    return channels[0:split], channels[split:]

################################
# Subsampling
################################

# 4:2:0 subsampling for YUV image
def subsampling(image):
    y = image[:, :, 0]
    u = image[:, :, 1]
    v = image[:, :, 2]

    # Keep only the top left sample of every 2x2 block for chrominance
    u = u[::2, ::2]
    v = v[::2, ::2]

    return y, u, v

# Scales up a channel that was previously subsampled
# Each pixel now corresponds to a 2x2 block of pixels
def scale(channel):
    return channel.repeat(2, axis=1).repeat(2, axis=0)

################################
# Discrete Cosine Transform
################################

# Creates the n-by-n DCT matrix
def create_dct_matrix(n):
    matrix = np.zeros(shape=(n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            a = math.sqrt(1 / n) if i == 0 else math.sqrt(2 / n)
            matrix[i][j] = a * math.cos(((2 * j + 1) * i * math.pi) / (2 * n))

    return matrix

# Computes the 2D DCT for each 8x8 submatrix of a matrix, in place
# Pass inverse=True to compute the inverse 2D DCT
def dct2d(matrix, inverse=False):
    dct_matrix = create_dct_matrix(8)
    dct_transpose_matrix = np.transpose(dct_matrix)
    (height, width) = matrix.shape

    for y in range(height // 8):
        for x in range(width // 8):
            block = matrix[8*y:8*y+8, 8*x:8*x+8]

            if inverse:
                block = np.matmul(dct_transpose_matrix, np.matmul(block, dct_matrix))
            else:
                block = np.matmul(dct_matrix, np.matmul(block, dct_transpose_matrix))

            matrix[8*y:8*y+8, 8*x:8*x+8] = block

    return matrix

def inverse_dct2d(matrix):
    return dct2d(matrix, True)

################################
# Quantization
################################

# Create the quantization matrix using the Q factor (quality)
# Uses the base matrix and formula from the IJG (Independent JPEG Group)
def create_quantization_matrix(quality):
    base_matrix = np.array([
      [16,  11,  10,  16,  24,  40,  51,  61],
      [12,  12,  14,  19,  26,  58,  60,  55],
      [14,  13,  16,  24,  40,  57,  69,  56],
      [14,  17,  22,  29,  51,  87,  80,  62],
      [18,  22,  37,  56,  68, 109, 103,  77],
      [24,  35,  55,  64,  81, 104, 113,  92],
      [49,  64,  78,  87, 103, 121, 120, 101],
      [72,  92,  95,  98, 112, 100, 103,  99],
    ])

    if quality < 50:
        s = 5000 / quality
    else:
        s = 200 - 2 * quality

    quantization_matrix = np.floor((s * base_matrix + 50) / 100)
    quantization_matrix[quantization_matrix == 0] = 1

    return quantization_matrix

# Applies quantization to each block using the quality factor
# Pass inverse=True to perform the inverse quantization
def quantization(matrix, quality, inverse=False):
    quantization_matrix = create_quantization_matrix(quality)
    (height, width) = matrix.shape

    for y in range(height // 8):
        for x in range(width // 8):
            block = matrix[8*y:8*y+8, 8*x:8*x+8]

            if inverse:
                block = np.round(block * quantization_matrix)
            else:
                block = np.round(block / quantization_matrix)

            matrix[8*y:8*y+8, 8*x:8*x+8] = block

    return matrix

def inverse_quantization(matrix, quality):
    return quantization(matrix, quality, True)

################################
# Runlength Coding
################################

# Get an array of index pairs to use for following a zigzag scan of a block
def get_zigzag_order(block_size):
    indices = np.indices((block_size, block_size)).transpose(1, 2, 0)
    flipped = np.fliplr(indices)
    order = np.zeros((block_size ** 2, 2), dtype=np.int64)
    upwards = True

    i = 0
    for offset in range(block_size - 1, -block_size, -1):
        diagonal = np.diagonal(flipped, offset=offset).transpose()

        if offset % 2 == 1:
            diagonal = np.flip(diagonal, axis=1)

        for coordinates in diagonal:
            order[i] = coordinates
            i += 1
    return order

# Perform runlength encoding for a channel
def runlength_encode(channel):
    zigzag_order = get_zigzag_order(8)

    (height, width) = channel.shape
    pairs = []

    for y in range(height // 8):
        for x in range(width // 8):
            block = channel[8*y:8*y+8, 8*x:8*x+8]
            skip = 0

            for [i, j] in zigzag_order:
                value = block[i, j]
                if value == 0:
                    skip += 1
                else:
                    pairs.append([skip, value])
                    skip = 0

            pairs.append([0, 0])

    return np.array(pairs, dtype=np.int16).flatten()

# Decode the runlength pairs and reconstruct the original channel
def runlength_decode(height, width, pairs):
    pairs = pairs.reshape((-1, 2))
    zigzag_order = get_zigzag_order(8)
    matrix = np.zeros((height, width))
    y_max = height // 8
    x_max = width // 8

    # Indices of current block
    y = 0
    x = 0

    # Index of position within zigzag scan
    index = 0

    for [skip, value] in pairs:
        if skip == 0 and value == 0:
            x += 1
            if x == x_max:
                y += 1
                x = 0
            index = 0
        else:
            index += skip
            i, j = zigzag_order[index]
            matrix[y*8+i, x*8+j] = value
            index += 1

    return matrix

################################
# Entropy Coding
################################

# A node of a Huffman tree
class TreeNode:
    def __init__(self, value=None):
        self.left = None
        self.right = None
        self.value = value

# Merge two equal size full binary trees
def merge_binary_tree(root_value, left_subtree, right_subtree):
    root = TreeNode(root_value)
    root.left = left_subtree
    root.right = right_subtree

    return root

# Find the frequency of each unique symbol in a given sequence
def find_frequencies(sequence, sequence_length):
    frequencies = {}
    for symbol in sequence:
        if not symbol in frequencies:
            frequencies[symbol] = 1
        else:
            frequencies[symbol] += 1

    return frequencies

# Create the Huffman tree from a given sequence of symbols
def create_huffman_tree(sequence, sequence_length):
    frequencies = find_frequencies(sequence, sequence_length)

    sorted = [(count, i, TreeNode(symbol)) for i, (symbol, count) in enumerate(frequencies.items())]
    heapq.heapify(sorted)
    next_parent_node = 1

    while (len(sorted) > 1):
        # Get the two subtrees with the lowest counts
        (count1, id1, left_subtree) = heapq.heappop(sorted)
        (count2, id2, right_subtree) = heapq.heappop(sorted)

        new_subtree = merge_binary_tree(None, left_subtree, right_subtree)
        next_parent_node += 1

        heapq.heappush(sorted, (count1 + count2, id1, new_subtree))

    _, _, huffman_tree = heapq.heappop(sorted)

    return huffman_tree

# Find all codewords of symbols in the Huffman tree
def find_codewords(huffman_tree):
    def recursive_codewords(tree_node, codewords, current_codeword):
        if tree_node.left is None or tree_node.right is None:
            codewords[tree_node.value] = current_codeword
            return

        recursive_codewords(tree_node.left, codewords, current_codeword + '0')
        recursive_codewords(tree_node.right, codewords, current_codeword + '1')
        return

    codewords = {}
    recursive_codewords(huffman_tree, codewords, '')
    return codewords

# Encodes a single node of a Huffman tree into a binary bitstring
def encode_huffman_node(node, stream):
    if node.left == None and node.right == None:
        stream.append('1' + np.binary_repr(node.value, width=12))
        return 1

    unique_codes = 0
    stream.append('0')

    unique_codes += encode_huffman_node(node.left, stream)
    unique_codes += encode_huffman_node(node.right, stream)

    return unique_codes

# Encodes a Huffman tree into a binary bitstring
def encode_huffman_tree(huffman_tree):
    stream = []
    unique_codes = encode_huffman_node(huffman_tree, stream)
    return unique_codes, ''.join(stream)

# Converts binary bitstring to signed integer
def binary_to_int(bitstring, width):
    number = int(bitstring, 2)
    if number > ((2 << (width - 2)) - 1):
        return ((2 << (width - 1)) - number) * (-1)
    else:
        return number

# Decode the binary bitstring for a single node of a Huffman tree
# to get the node's original value
def decode_huffman_node(node, stream, index):
    if stream[index] == '1':
        index += 1
        node.left = None
        node.right = None
        node.value = binary_to_int(stream[index:index+12], 12)
        index += 12
        return index

    index += 1
    left_node = TreeNode()
    right_node = TreeNode()
    node.left = left_node
    node.right = right_node
    index = decode_huffman_node(left_node, stream, index)
    index = decode_huffman_node(right_node, stream, index)

    return index

# Decode the binary bitstring of an encoded Huffman tree
# to reconstruct the original tree
def decode_huffman_tree(stream):
    index = 0
    tree = TreeNode()
    decode_huffman_node(tree, stream, index)
    return tree

# Get the codeword for a given symbol
def get_codeword(value, codewords):
    return codewords[value]

# Use Huffman coding to encode an array of values into a bitstring
# Returns a binary bitstring of the Huffman tree, and the compressed values
def huffman_encode(values):
    huffman_tree = create_huffman_tree(values, len(values))
    codewords = find_codewords(huffman_tree)
    get_codeword = np.vectorize(codewords.__getitem__)
    values = get_codeword(values)
    unique_codes, tree_stream = encode_huffman_tree(huffman_tree)
    value_stream = ''.join(values)
    return unique_codes, tree_stream, value_stream

# Inverts a dictionary, making the values the keys and the keys the values
def invert_dictionary(dictionary):
    return {value: key for key, value in dictionary.items()}

# Use Huffman coding to decode a binary bitstring of compressed values
def huffman_decode(tree_stream, value_stream):
    huffman_tree = decode_huffman_tree(tree_stream)
    decodewords = invert_dictionary(find_codewords(huffman_tree))

    index = 0
    length = 1
    decoded = []
    while index < len(value_stream):
        slice = value_stream[index:index+length]
        if slice in decodewords:
            decoded.append(decodewords[slice])
            index += length
            length = 1
        else:
            length += 1

    return np.array(decoded, dtype=np.int16)

################################
# Compression and Decompression
################################

# Compress the image
def compress(q_factor, image):
    (height, width, _) = image.shape

    yuv_image = rgb_image_to_yuv(image)

    y_channel, u_channel, v_channel = subsampling(yuv_image)

    # Pad each channel so that height and width are multiples of 8
    padded_y_channel = pad_edges(y_channel)
    padded_u_channel = pad_edges(u_channel)
    padded_v_channel = pad_edges(v_channel)

    (padded_y_height, padded_y_width) = padded_y_channel.shape
    (padded_u_height, padded_u_width) = padded_u_channel.shape
    (padded_v_height, padded_v_width) = padded_v_channel.shape

    # Apply discrete cosine transform
    dct_y_channel = dct2d(padded_y_channel)
    dct_u_channel = dct2d(padded_u_channel)
    dct_v_channel = dct2d(padded_v_channel)

    # Apply quantization
    quantized_y_channel = quantization(dct_y_channel, q_factor)
    quantized_u_channel = quantization(dct_u_channel, q_factor)
    quantized_v_channel = quantization(dct_v_channel, q_factor)

    # Apply runlength encoding
    runlength_y_channel = runlength_encode(quantized_y_channel)
    runlength_u_channel = runlength_encode(quantized_u_channel)
    runlength_v_channel = runlength_encode(quantized_v_channel)
    runlength_uv_channels = combine_channels(runlength_u_channel, runlength_v_channel)

    # Apply Huffman encoding
    _, y_tree, y_values = huffman_encode(runlength_y_channel)
    _, uv_tree, uv_values = huffman_encode(runlength_uv_channels)

    return height, width, y_tree, uv_tree, y_values, uv_values

# Decompress the image
def decompress(q_factor, height, width, y_tree, uv_tree, y_values, uv_values):
    padded_y_height = math.ceil(height / 8) * 8
    padded_y_width = math.ceil(width / 8) * 8
    padded_u_height = math.ceil(math.ceil(height / 2) / 8) * 8
    padded_u_width = math.ceil(math.ceil(width / 2) / 8) * 8
    padded_v_height = padded_u_height
    padded_v_width = padded_u_width

    # Apply Huffman decoding
    entropy_y_channel = huffman_decode(y_tree, y_values)
    entropy_uv_channels = huffman_decode(uv_tree, uv_values)

    # Apply runlength decoding
    decoded_y_channel = runlength_decode(padded_y_height, padded_y_width, entropy_y_channel)
    decoded_uv_channels = runlength_decode(padded_u_height * 2, padded_u_width, entropy_uv_channels)
    decoded_u_channel, decoded_v_channel = split_channels(decoded_uv_channels)

    # Apply inverse quantization
    unquantized_y_channel = inverse_quantization(decoded_y_channel, q_factor)
    unquantized_u_channel = inverse_quantization(decoded_u_channel, q_factor)
    unquantized_v_channel = inverse_quantization(decoded_v_channel, q_factor)

    # Apply inverse discrete cosine transform
    undct_y_channel = inverse_dct2d(unquantized_y_channel)
    undct_u_channel = inverse_dct2d(unquantized_u_channel)
    undct_v_channel = inverse_dct2d(unquantized_v_channel)

    y_channel = unpad_edges(undct_y_channel, height, width)
    u_channel = unpad_edges(scale(unpad_edges(undct_u_channel, math.ceil(height / 2), math.ceil(width / 2))), height, width)
    v_channel = unpad_edges(scale(unpad_edges(undct_v_channel, math.ceil(height / 2), math.ceil(width / 2))), height, width)

    yuv_image = create_image_from_channels(y_channel, u_channel, v_channel)

    image = yuv_image_to_rgb(yuv_image)

    return image

################################
# Peak Signal to Noise Ratio
################################
# Uses the formula from https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def calculate_psnr(original_image, new_image):
    max_pixel = 255
    (m, n, _) = original_image.shape

    squared_errors = (original_image - new_image) ** 2
    mse = np.sum(squared_errors) / (m * n * 3)

    db = 20 * math.log10(max_pixel) - 10 * math.log10(mse)

    return db

################################
# Read and Save Files
################################

# Returns the file name to read
def get_file_name():
    return tk.filedialog.askopenfilename()

# Returns the file extension of a file name
def get_file_extension(file_name):
    return file_name[file_name.rindex('.'):]

# Returns the file name without the extension
def get_file_name_without_extension(file_name):
    return file_name[:file_name.rindex('.')]

# Returns the file size of an open file
def get_file_size(file):
    return os.fstat(f.fileno()).st_size

# Returns the size of a file
def get_file_size(file_name):
    return os.stat(file_name).st_size

# Reads the bitmap file
# Returns a numpy array of the pixels with shape (height, width, 3)
def read_bitmap_file(file_name):
    file = open(file_name, 'rb')

    # Check that the file is a bitmap file
    signature = file.read(2).decode('ascii') # Get Signature
    if signature != 'BM':
        sys.exit('ERROR: {} is not a bitmap file'.format(file_name))

    file_size = int.from_bytes(file.read(4), 'little') # Get FileSize
    file.read(4) # Discard reserved
    data_offset = int.from_bytes(file.read(4), 'little') # Get DataOffset

    file.read(4) # Discard Size
    width = int.from_bytes(file.read(4), 'little') # Get Width
    height = int.from_bytes(file.read(4), 'little') # Get Height
    file.read(2) # Discard Planes

    # Check that the file is 24 bits per pixel
    bits_per_pixel = int.from_bytes(file.read(2), 'little') # Get BitsPerPixel
    if bits_per_pixel != 24:
        sys.exit('ERROR: file must have 24 bits per pixel')

    # Check that the file is uncompressed
    compression = int.from_bytes(file.read(4), 'little') # Get Compression
    if compression != 0:
        sys.exit('ERROR: file must be uncompressed')

    # Discard the rest of the header
    file.read(data_offset - 34)

    # Create numpy array to hold pixel values
    image = np.zeros(shape=(height, width, 3), dtype=np.float64)

    # Coordinates of current pixel being read/drawn
    x = 0
    y = height - 1

    # Compute the number of padding bytes at the end of each row
    row_padding = (math.ceil(width * 3 / 4) * 4) - (width * 3)

    # Read the full image from the file
    while True:
        # Read each channel value of the next pixel
        blue = file.read(1)
        green = file.read(1)
        red = file.read(1)

        # End of file reached
        if not (blue and green and red):
            break

        # Convert bytes to integers
        blue = int.from_bytes(blue, 'little')
        green = int.from_bytes(green, 'little')
        red = int.from_bytes(red, 'little')

        # Save pixel in image array
        image[y][x] = (red, green, blue)

        x += 1
        if x == width:
            x = 0
            y -= 1
            file.read(row_padding) # Discard padding at end of row

    file.close()
    return image

# Adds padding bits to the front of a bitstring
# Makes the bitstring evenly divisible into bytes
def pad_bits(bitstring):
    length = len(bitstring)
    padding = 8 - (length % 8)
    padstring = '0' * (padding - 1) + '1'
    return padstring + bitstring

# Saves the compressed image as a .IMG file
def save_img_file(file_name, quality, height, width, y_tree, uv_tree, y_values, uv_values):
    file = open(file_name + '.IMG', 'wb')
    file.write(bytearray('IMG', encoding='ascii'))
    file.write(quality.to_bytes(1, byteorder='big', signed=False))
    file.write(height.to_bytes(2, byteorder='big', signed=False))
    file.write(width.to_bytes(2, byteorder='big', signed=False))

    y_tree = pad_bits(y_tree)
    uv_tree = pad_bits(uv_tree)
    y_values = pad_bits(y_values)
    uv_values = pad_bits(uv_values)

    y_tree_size = len(y_tree) // 8
    uv_tree_size = len(uv_tree) // 8
    y_values_size = len(y_values) // 8
    uv_values_size = len(uv_values) // 8

    file.write(y_tree_size.to_bytes(2, byteorder='big', signed=False))
    file.write(uv_tree_size.to_bytes(2, byteorder='big', signed=False))
    file.write(y_values_size.to_bytes(4, byteorder='big', signed=False))
    file.write(uv_values_size.to_bytes(4, byteorder='big', signed=False))

    y_tree_bytes = [int(y_tree[i:i+8], 2) for i in range(0, len(y_tree), 8)]
    uv_tree_bytes = [int(uv_tree[i:i+8], 2) for i in range(0, len(uv_tree), 8)]
    y_values_bytes = [int(y_values[i:i+8], 2) for i in range(0, len(y_values), 8)]
    uv_values_bytes = [int(uv_values[i:i+8], 2) for i in range(0, len(uv_values), 8)]

    file.write(bytearray(y_tree_bytes))
    file.write(bytearray(uv_tree_bytes))
    file.write(bytearray(y_values_bytes))
    file.write(bytearray(uv_values_bytes))

    file.close()

    return

# Reads length number of bytes from file and converts them to a binary bitstring
# Also removes any padding at the front of the bitstring
def read_bitstring(file, length):
    values = []
    for i in range(length):
        values.append('{:08b}'.format(int.from_bytes(file.read(1), 'big', signed=False)))
    bitstring = ''.join(values)
    padding_index = bitstring.index('1')
    return bitstring[padding_index+1:]

# Reads the IMG file
# Returns a numpy array of the pixels with shape (height, width, 3)
def read_img_file(file_name):
    file = open(file_name, 'rb')

    # Check that the file is an IMG file
    signature = file.read(3).decode('ascii') # Get Signature
    if signature != 'IMG':
        sys.exit('ERROR: {} is not an IMG file'.format(file_name))

    quality = int.from_bytes(file.read(1), 'big', signed=False) # Get quality factor

    height = int.from_bytes(file.read(2), 'big', signed=False) # Get image height
    width = int.from_bytes(file.read(2), 'big', signed=False) # Get image width

    y_tree_size = int.from_bytes(file.read(2), 'big', signed=False) # Get Y Huffman tree size
    uv_tree_size = int.from_bytes(file.read(2), 'big', signed=False) # Get Y Huffman tree size
    y_values_size = int.from_bytes(file.read(4), 'big', signed=False) # Get Y Huffman tree size
    uv_values_size = int.from_bytes(file.read(4), 'big', signed=False) # Get Y Huffman tree size

    y_tree = read_bitstring(file, y_tree_size)
    uv_tree = read_bitstring(file, uv_tree_size)
    y_values = read_bitstring(file, y_values_size)
    uv_values = read_bitstring(file, uv_values_size)

    file.close()

    return quality, height, width, y_tree, uv_tree, y_values, uv_values

################################
# Main Program
################################

# The initial function called for the program
def main():
    # Create the display window
    window = tk.Tk()
    window.title('Loading...')

    # Get the file and read it
    file_name = get_file_name()

    file_extension = get_file_extension(file_name)

    if file_extension == '.bmp':
        # Read the file
        bitmap_image = read_bitmap_file(file_name)
        (height, width, _) = bitmap_image.shape

        # Get the quality factor from the user
        quality = tk.simpledialog.askstring(title='Quality Factor', prompt='Enter a quality factor (integer from 1-100): ')

        try:
            int(quality)
        except ValueError:
            print('ERROR: not a valid quality factor')

        quality = int(quality)

        if quality < 1 or quality > 100:
            print('ERROR: quality factor must be between 1 and 100')

        file_name_no_extension = get_file_name_without_extension(file_name)

        # Compress the image
        compress_start_time = time.time()
        height, width, y_tree, uv_tree, y_values, uv_values = compress(quality, bitmap_image)
        compress_end_time = time.time()
        print('Compression time: ', compress_end_time - compress_start_time, 's')

        # Save the compressed image
        save_img_file(file_name_no_extension, quality, height, width, y_tree, uv_tree, y_values, uv_values)

        # Decompress the compressed image
        image = decompress(quality, height, width, y_tree, uv_tree, y_values, uv_values)
        image = image.astype(np.uint8)
        bitmap_image = bitmap_image.astype(np.uint8)

        buffer_width = 64
        buffer_height = 64

        # Compute compression ratio and PSNR
        original_size = get_file_size(file_name_no_extension + '.bmp')
        compressed_size = get_file_size(file_name_no_extension + '.IMG')
        compression_ratio = format(original_size / compressed_size, '.2f')
        psnr = format(calculate_psnr(bitmap_image, image), '.2f')

        # Display the original image and IMG image
        canvas = tk.Canvas(window, width=width*2+buffer_width, height=height+2*buffer_height)
        canvas_bmp_image = draw_image(width, height, bitmap_image)
        canvas_img_image = draw_image(width, height, image)
        canvas.create_image(0, 0, image=canvas_bmp_image, state='normal', anchor='nw')
        canvas.create_image(width+buffer_width, 0, image=canvas_img_image, state='normal', anchor='nw')
        original_text = tk.Label(text='BMP image', font=tk.font.Font(size=18))
        compressed_text = tk.Label(text='IMG image', font=tk.font.Font(size=18))
        ratio_text = tk.Label(text='Compression ratio: ' + compression_ratio, font=tk.font.Font(size=18))
        psnr_text = tk.Label(text='PSNR: ' + psnr + ' dB', font=tk.font.Font(size=18))
        print('Compression ratio: ' + compression_ratio)
        print('PSNR', psnr, 'dB')

        original_text.pack()
        original_text.place(x=0, y=height)
        compressed_text.pack()
        compressed_text.place(x=width+buffer_width, y=height)
        ratio_text.pack()
        ratio_text.place(x=0, y=height+buffer_height//2)
        psnr_text.pack()
        psnr_text.place(x=0, y=height+buffer_height)
        canvas.pack()

    elif file_extension == '.IMG':
        # Read the compressed file
        quality, height, width, y_tree, uv_tree, y_values, uv_values = read_img_file(file_name)

        # Decompress the image
        decompress_start_time = time.time()
        image = decompress(quality, height, width, y_tree, uv_tree, y_values, uv_values)
        image = image.astype(np.uint8)
        decompress_end_time = time.time()
        print('Decompression time: ', decompress_end_time - decompress_start_time, 's')

        # Display the uncompressed image
        canvas = tk.Canvas(window, width=width, height=height)
        canvas_image = draw_image(width, height, image)
        canvas.create_image(0, 0, image=canvas_image, state='normal', anchor='nw')
        canvas.pack()

    else:
        print('ERROR: {} is not a supported file type'.format(file_extension))

    window.title(os.path.basename(file_name))

    # Update the window
    window.mainloop()

    return

if __name__ == '__main__':
    main()
