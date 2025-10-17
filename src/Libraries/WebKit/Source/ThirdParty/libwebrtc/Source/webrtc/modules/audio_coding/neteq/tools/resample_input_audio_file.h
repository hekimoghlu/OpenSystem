/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_RESAMPLE_INPUT_AUDIO_FILE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_RESAMPLE_INPUT_AUDIO_FILE_H_

#include <string>

#include "absl/strings/string_view.h"
#include "common_audio/resampler/include/resampler.h"
#include "modules/audio_coding/neteq/tools/input_audio_file.h"

namespace webrtc {
namespace test {

// Class for handling a looping input audio file with resampling.
class ResampleInputAudioFile : public InputAudioFile {
 public:
  ResampleInputAudioFile(absl::string_view file_name,
                         int file_rate_hz,
                         bool loop_at_end = true)
      : InputAudioFile(file_name, loop_at_end),
        file_rate_hz_(file_rate_hz),
        output_rate_hz_(-1) {}
  ResampleInputAudioFile(absl::string_view file_name,
                         int file_rate_hz,
                         int output_rate_hz,
                         bool loop_at_end = true)
      : InputAudioFile(file_name, loop_at_end),
        file_rate_hz_(file_rate_hz),
        output_rate_hz_(output_rate_hz) {}

  ResampleInputAudioFile(const ResampleInputAudioFile&) = delete;
  ResampleInputAudioFile& operator=(const ResampleInputAudioFile&) = delete;

  bool Read(size_t samples, int output_rate_hz, int16_t* destination);
  bool Read(size_t samples, int16_t* destination) override;
  void set_output_rate_hz(int rate_hz);

 private:
  const int file_rate_hz_;
  int output_rate_hz_;
  Resampler resampler_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_RESAMPLE_INPUT_AUDIO_FILE_H_
