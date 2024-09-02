import numpy as np

def save_file(spec, rms):
    """
    spec: 2D array in numpy
    rms: float (The RMS value of the difference)

    Save the difference to a file
    The goal is to analyze the difference between the two files.
    """
    with open("Compared_Data.txt", "w") as f:
        for i in range(len(spec)):
            for j in range(len(spec[i])):
                f.write(str(spec[i][j]) + " ")
            f.write("\n")
        f.write("RMS: " + str(rms))

# Read the first file
file1 = 'mel_spectrogram_segment_1.txt'
with open(file1, 'r') as f1:
    lines1 = f1.readlines()

# Read the second file
file2 = 'python_data.txt'
with open(file2, 'r') as f2:
    lines2 = f2.readlines()

# Convert the values to a 2D array
array1 = np.array([list(map(float, line.strip().split(' '))) for line in lines1])
array2 = np.array([list(map(float, line.strip().split(' '))) for line in lines2])

# Calculate the difference
diff = array2 - array1

# Show the difference

# Calculate the RMS
rms = np.sqrt(np.mean(diff**2))
print("RMS: ", rms)

# Set the printing threshold to the size of the array
np.set_printoptions( threshold= 128*216 ) 
save_file(diff, rms) # Save the difference to a file