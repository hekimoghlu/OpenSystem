/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef MODULES_AUDIO_PROCESSING_TEST_TEST_UTILS_H_
#define MODULES_AUDIO_PROCESSING_TEST_TEST_UTILS_H_

#include <math.h>

#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/audio/audio_processing.h"
#include "common_audio/channel_buffer.h"
#include "common_audio/wav_file.h"

namespace webrtc {

static const AudioProcessing::Error kNoErr = AudioProcessing::kNoError;
#define EXPECT_NOERR(expr) EXPECT_EQ(kNoErr, (expr))

// Encapsulates samples and metadata for an integer frame.
struct Int16FrameData {
  // Max data size that matches the data size of the AudioFrame class, providing
  // storage for 8 channels of 96 kHz data.
  static const int kMaxDataSizeSamples = 7680;

  Int16FrameData() {
    sample_rate_hz = 0;
    num_channels = 0;
    samples_per_channel = 0;
    data.fill(0);
  }

  void CopyFrom(const Int16FrameData& src) {
    samples_per_channel = src.samples_per_channel;
    sample_rate_hz = src.sample_rate_hz;
    num_channels = src.num_channels;

    const size_t length = samples_per_channel * num_channels;
    RTC_CHECK_LE(length, kMaxDataSizeSamples);
    memcpy(data.data(), src.data.data(), sizeof(int16_t) * length);
  }
  std::array<int16_t, kMaxDataSizeSamples> data;
  int32_t sample_rate_hz;
  size_t num_channels;
  size_t samples_per_channel;
};

// Reads ChannelBuffers from a provided WavReader.
class ChannelBufferWavReader final {
 public:
  explicit ChannelBufferWavReader(std::unique_ptr<WavReader> file);
  ~ChannelBufferWavReader();

  ChannelBufferWavReader(const ChannelBufferWavReader&) = delete;
  ChannelBufferWavReader& operator=(const ChannelBufferWavReader&) = delete;

  // Reads data from the file according to the `buffer` format. Returns false if
  // a full buffer can't be read from the file.
  bool Read(ChannelBuffer<float>* buffer);

 private:
  std::unique_ptr<WavReader> file_;
  std::vector<float> interleaved_;
};

// Writes ChannelBuffers to a provided WavWriter.
class ChannelBufferWavWriter final {
 public:
  explicit ChannelBufferWavWriter(std::unique_ptr<WavWriter> file);
  ~ChannelBufferWavWriter();

  ChannelBufferWavWriter(const ChannelBufferWavWriter&) = delete;
  ChannelBufferWavWriter& operator=(const ChannelBufferWavWriter&) = delete;

  void Write(const ChannelBuffer<float>& buffer);

 private:
  std::unique_ptr<WavWriter> file_;
  std::vector<float> interleaved_;
};

// Takes a pointer to a vector. Allows appending the samples of channel buffers
// to the given vector, by interleaving the samples and converting them to float
// S16.
class ChannelBufferVectorWriter final {
 public:
  explicit ChannelBufferVectorWriter(std::vector<float>* output);
  ChannelBufferVectorWriter(const ChannelBufferVectorWriter&) = delete;
  ChannelBufferVectorWriter& operator=(const ChannelBufferVectorWriter&) =
      delete;
  ~ChannelBufferVectorWriter();

  // Creates an interleaved copy of `buffer`, converts the samples to float S16
  // and appends the result to output_.
  void Write(const ChannelBuffer<float>& buffer);

 private:
  std::vector<float> interleaved_buffer_;
  std::vector<float>* output_;
};

// Exits on failure; do not use in unit tests.
FILE* OpenFile(absl::string_view filename, absl::string_view mode);

void SetFrameSampleRate(Int16FrameData* frame, int sample_rate_hz);

template <typename T>
void SetContainerFormat(int sample_rate_hz,
                        size_t num_channels,
                        Int16FrameData* frame,
                        std::unique_ptr<ChannelBuffer<T> >* cb) {
  SetFrameSampleRate(frame, sample_rate_hz);
  frame->num_channels = num_channels;
  cb->reset(new ChannelBuffer<T>(frame->samples_per_channel, num_channels));
}

template <typename T>
float ComputeSNR(const T* ref, const T* test, size_t length, float* variance) {
  float mse = 0;
  float mean = 0;
  *variance = 0;
  for (size_t i = 0; i < length; ++i) {
    T error = ref[i] - test[i];
    mse += error * error;
    *variance += ref[i] * ref[i];
    mean += ref[i];
  }
  mse /= length;
  *variance /= length;
  mean /= length;
  *variance -= mean * mean;

  float snr = 100;  // We assign 100 dB to the zero-error case.
  if (mse > 0)
    snr = 10 * log10(*variance / mse);
  return snr;
}

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_TEST_UTILS_H_
