CC=g++
LDFLAGS+= -lboost_system -lboost_filesystem -lboost_iostreams -lsndfile -lfftw3 -lsamplerate
CPP = spectrogram_with_fftw3.cpp spectrogram.cpp
OBJ1 = spectrogram_with_fftw3.o
OBJ2 = spectrogram.o
LIB = /librosa

all: fft spectrogram

fft: $(OBJ1)
	$(CC) -o fft $(OBJ1) $(LDFLAGS)

spectrogram: $(OBJ2) 
	$(CC) -o spectrogram -I $(LIB) $(OBJ2) $(LDFLAGS)
	
%.o: %.cpp
	$(CC) -c $< -o $@

clean:
	rm -f *.o fft spectrogram
