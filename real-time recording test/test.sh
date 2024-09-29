#!/bin/bash

WATCH_DIR="~/Desktop/test2"
BASE_NAME="temp_audio"
EXTENSION=".wav"
COUNTER=1

while true; do
   	# 構造檔案名稱
	FILE_NAME="${BASE_NAME}${COUNTER}${EXTENSION}"
		# 檢查檔案是否存在
	if [ -f "$FILE_NAME" ]; then
		echo "File $FILE_NAME detected，start calculating..."

        	# 在這裡執行你的運算
        	./cal_mel_spec $FILE_NAME
		./plot_mel_spec mel_spectrogram_segment_1.txt mel_spectrogram_segment_1.png

        	echo "Finish calculation, remove the file $FILE_NAME..."
        	rm "$FILE_NAME"

        	# 更新計數器以檢查下一個檔案
        	COUNTER=$((COUNTER + 1))
    	fi

    	sleep 5  # 每隔 5 秒檢查一次
done

