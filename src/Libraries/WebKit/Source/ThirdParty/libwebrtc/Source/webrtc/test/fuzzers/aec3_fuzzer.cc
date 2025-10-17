/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include <optional>

#include "api/audio/audio_processing.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"
#include "modules/audio_processing/audio_buffer.h"
#include "test/fuzzers/fuzz_data_helper.h"

namespace webrtc {
namespace {
using SampleRate = ::webrtc::AudioProcessing::NativeRate;

void PrepareAudioBuffer(int sample_rate_hz,
                        test::FuzzDataHelper* fuzz_data,
                        AudioBuffer* buffer) {
  float* const* channels = buffer->channels_f();
  for (size_t i = 0; i < buffer->num_channels(); ++i) {
    for (size_t j = 0; j < buffer->num_frames(); ++j) {
      channels[i][j] =
          static_cast<float>(fuzz_data->ReadOrDefaultValue<int16_t>(0));
    }
  }
  if (sample_rate_hz == 32000 || sample_rate_hz == 48000) {
    buffer->SplitIntoFrequencyBands();
  }
}

}  // namespace

void FuzzOneInput(const uint8_t* data, size_t size) {
  if (size > 200000) {
    return;
  }

  test::FuzzDataHelper fuzz_data(rtc::ArrayView<const uint8_t>(data, size));

  constexpr int kSampleRates[] = {16000, 32000, 48000};
  const int sample_rate_hz =
      static_cast<size_t>(fuzz_data.SelectOneOf(kSampleRates));

  constexpr int kMaxNumChannels = 9;
  const size_t num_render_channels =
      1 + fuzz_data.ReadOrDefaultValue<uint8_t>(0) % (kMaxNumChannels - 1);
  const size_t num_capture_channels =
      1 + fuzz_data.ReadOrDefaultValue<uint8_t>(0) % (kMaxNumChannels - 1);

  EchoCanceller3 aec3(EchoCanceller3Config(),
                      /*multichannel_config=*/std::nullopt, sample_rate_hz,
                      num_render_channels, num_capture_channels);

  AudioBuffer capture_audio(sample_rate_hz, num_capture_channels,
                            sample_rate_hz, num_capture_channels,
                            sample_rate_hz, num_capture_channels);
  AudioBuffer render_audio(sample_rate_hz, num_render_channels, sample_rate_hz,
                           num_render_channels, sample_rate_hz,
                           num_render_channels);

  // Fuzz frames while there is still fuzzer data.
  while (fuzz_data.BytesLeft() > 0) {
    bool is_capture = fuzz_data.ReadOrDefaultValue(true);
    bool level_changed = fuzz_data.ReadOrDefaultValue(true);
    if (is_capture) {
      PrepareAudioBuffer(sample_rate_hz, &fuzz_data, &capture_audio);
      aec3.ProcessCapture(&capture_audio, level_changed);
    } else {
      PrepareAudioBuffer(sample_rate_hz, &fuzz_data, &render_audio);
      aec3.AnalyzeRender(&render_audio);
    }
  }
}
}  // namespace webrtc
