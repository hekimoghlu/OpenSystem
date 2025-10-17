/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_INPUT_AUDIO_FILE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_INPUT_AUDIO_FILE_H_

#include <stdio.h>

#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace test {

// Class for handling a looping input audio file.
class InputAudioFile {
 public:
  explicit InputAudioFile(absl::string_view file_name, bool loop_at_end = true);

  virtual ~InputAudioFile();

  InputAudioFile(const InputAudioFile&) = delete;
  InputAudioFile& operator=(const InputAudioFile&) = delete;

  // Reads `samples` elements from source file to `destination`. Returns true
  // if the read was successful, otherwise false. If the file end is reached,
  // the file is rewound and reading continues from the beginning.
  // The output `destination` must have the capacity to hold `samples` elements.
  virtual bool Read(size_t samples, int16_t* destination);

  // Fast-forwards (`samples` > 0) or -backwards (`samples` < 0) the file by the
  // indicated number of samples. Just like Read(), Seek() starts over at the
  // beginning of the file if the end is reached. However, seeking backwards
  // past the beginning of the file is not possible.
  virtual bool Seek(int samples);

  // Creates a multi-channel signal from a mono signal. Each sample is repeated
  // `channels` times to create an interleaved multi-channel signal where all
  // channels are identical. The output `destination` must have the capacity to
  // hold samples * channels elements. Note that `source` and `destination` can
  // be the same array (i.e., point to the same address).
  static void DuplicateInterleaved(const int16_t* source,
                                   size_t samples,
                                   size_t channels,
                                   int16_t* destination);

 private:
  FILE* fp_;
  const bool loop_at_end_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_INPUT_AUDIO_FILE_H_
