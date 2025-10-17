/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_OUTPUT_WAV_FILE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_OUTPUT_WAV_FILE_H_

#include <string>

#include "absl/strings/string_view.h"
#include "common_audio/wav_file.h"
#include "modules/audio_coding/neteq/tools/audio_sink.h"

namespace webrtc {
namespace test {

class OutputWavFile : public AudioSink {
 public:
  // Creates an OutputWavFile, opening a file named `file_name` for writing.
  // The output file is a PCM encoded wav file.
  OutputWavFile(absl::string_view file_name,
                int sample_rate_hz,
                int num_channels = 1)
      : wav_writer_(file_name, sample_rate_hz, num_channels) {}

  OutputWavFile(const OutputWavFile&) = delete;
  OutputWavFile& operator=(const OutputWavFile&) = delete;

  bool WriteArray(const int16_t* audio, size_t num_samples) override {
    wav_writer_.WriteSamples(audio, num_samples);
    return true;
  }

 private:
  WavWriter wav_writer_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_OUTPUT_WAV_FILE_H_
