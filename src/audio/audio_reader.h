
//
// Created by Yin, Weisu on 1/6/21.
//

#ifndef DECORD_AUDIO_READER_H_
#define DECORD_AUDIO_READER_H_

#include <vector>

#include "../../include/decord/audio_interface.h"

// Include AVChannelLayout
#ifdef __cplusplus
extern "C" {
#endif
#include <libavutil/channel_layout.h>
#ifdef __cplusplus
}
#endif

namespace decord {

class AudioReader : public AudioReaderInterface {
public:
    AudioReader(std::string fn, int sampleRate, DLContext ctx, int io_type=kNormal, bool mono=true);
    ~AudioReader();
    NDArray GetNDArray();
    int GetNumPaddingSamples();
    double GetDuration();
    int64_t GetNumSamplesPerChannel();
    int GetNumChannels();
    void GetInfo();
private:
    int Decode(std::string fn, int io_type);
    void DecodePacket(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, int streamIndex);
    void HandleFrame(AVCodecContext *pCodecContext, AVFrame *pFrame);
    void DrainDecoder(AVCodecContext *pCodecContext, AVFrame *pFrame);
    void InitSWR(AVCodecContext *pCodecContext);
    void ToNDArray();
    void SaveToVector(float** buffer, int numChannels, int numSamples);

    DLContext ctx;
    std::unique_ptr<ffmpeg::AVIOBytesContext> io_ctx_;  // avio context for raw memory access
    AVFormatContext *pFormatContext = nullptr;
    struct SwrContext* swr = nullptr;
    // AVCodec* pCodec;  // Removed: No longer needed
    AVCodecParameters* pCodecParameters = nullptr;
    AVCodecContext * pCodecContext = nullptr;
    int audioStreamIndex = -1;
    // std::vector<std::unique_ptr<AudioStream>> audios; // No longer needed.
    std::vector<std::vector<float>> outputVector;
    NDArray output;
    // padding is the start time in seconds of the first audio sample
    double padding = -1.0;
    std::string filename;
    int originalSampleRate = 0;
    int targetSampleRate = -1;
    int numChannels = 0;
    bool mono;
    int64_t totalSamplesPerChannel = 0;
    int64_t totalConvertedSamplesPerChannel = 0;
    double timeBase = 0.0;
    double duration = 0.0;
};

}  // namespace decord
#endif  // DECORD_AUDIO_READER_H_
