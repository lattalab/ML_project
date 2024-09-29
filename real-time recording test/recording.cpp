#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>

using namespace std;

// 記錄音時間，單位是秒
const int RECORDING_INTERVAL = 5;
// seqence number of audio file
int count = 1; 

void recordAudio() {
    // GStreamer 初始化
    gst_init(nullptr, nullptr);

    while (true) {
        string filename = "temp_audio" + to_string(count++) + ".wav";
        // 創建 GStreamer 錄音管道
        string pipeline_desc =
            "alsasrc ! "
            "audioconvert ! "
            "audioresample ! "
            "audio/x-raw,channels=1,rate=22050 ! "
            "wavenc ! "
            "filesink location=" + filename;

        GError *error = nullptr;
        GstElement *pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
        if (error) {
            cerr << "Failed to create pipeline: " << error->message << endl;
            g_error_free(error);
            return;
        }

        // 啟動管道
        gst_element_set_state(pipeline, GST_STATE_PLAYING);

        // 每隔 RECORDING_INTERVAL 秒寫入檔案
        cout << "Recording for " << RECORDING_INTERVAL << " seconds..." << endl;

        // 等待指定的時間
        std::this_thread::sleep_for(std::chrono::seconds(RECORDING_INTERVAL));

        // 在這裡可以將 temp_audio.wav 重命名或處理
        // 例如：將檔案名改為 temp_audio_1.wav，temp_audio_2.wav 等
        // 這裡簡單展示，只是顯示訊息
        cout << "Recorded 5 seconds of audio." << endl;

        // 清理
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }
    gst_deinit();
}

int main(int argc, char *argv[]) {
    recordAudio();
    return 0;
}
