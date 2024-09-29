CC = g++
CFLAGS = `pkg-config --cflags gstreamer-1.0`
LIBS = `pkg-config --libs gstreamer-1.0`
TARGET = record_audio
SRC = recording.cpp

all:
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)
