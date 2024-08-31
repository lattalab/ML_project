# 指定編譯器
CC = g++

# 指定編譯標誌
CXXFLAGS = `pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-fft-1.0 cairo`
# 指定鏈接標誌
LDFLAGS = `pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-fft-1.0 cairo`

# 檔案與目標
CPP = cal_mel_spec.cpp
EXE = cal_mel_spec
PLOT = plot_mel_spec

# 編譯目標
all: $(EXE) $(PLOT)

$(EXE): $(CPP)
	$(CC) $(CXXFLAGS) -o $(EXE) $(CPP) $(LDFLAGS)

$(PLOT): plot_mel_spec_from_txt.cpp
	$(CC) $(CXXFLAGS) -o $(PLOT) plot_mel_spec_from_txt.cpp $(LDFLAGS)

clean:
	rm -f $(EXE) $(PLOT)

