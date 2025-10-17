/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_OUTPUT_AUDIO_FILE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_OUTPUT_AUDIO_FILE_H_

#include <stdio.h>

#include <string>

#include "absl/strings/string_view.h"
#include "modules/audio_coding/neteq/tools/audio_sink.h"

namespace webrtc {
namespace test {

class OutputAudioFile : public AudioSink {
 public:
  // Creates an OutputAudioFile, opening a file named `file_name` for writing.
  // The file format is 16-bit signed host-endian PCM.
  explicit OutputAudioFile(absl::string_view file_name) {
    out_file_ = fopen(std::string(file_name).c_str(), "wb");
  }

  virtual ~OutputAudioFile() {
    if (out_file_)
      fclose(out_file_);
  }

  OutputAudioFile(const OutputAudioFile&) = delete;
  OutputAudioFile& operator=(const OutputAudioFile&) = delete;

  bool WriteArray(const int16_t* audio, size_t num_samples) override {
    RTC_DCHECK(out_file_);
    return fwrite(audio, sizeof(*audio), num_samples, out_file_) == num_samples;
  }

 private:
  FILE* out_file_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_OUTPUT_AUDIO_FILE_H_
