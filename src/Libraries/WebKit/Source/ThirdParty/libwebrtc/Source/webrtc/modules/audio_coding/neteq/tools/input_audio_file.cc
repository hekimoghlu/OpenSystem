/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#include "modules/audio_coding/neteq/tools/input_audio_file.h"

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

InputAudioFile::InputAudioFile(absl::string_view file_name, bool loop_at_end)
    : loop_at_end_(loop_at_end) {
  fp_ = fopen(std::string(file_name).c_str(), "rb");
  RTC_DCHECK(fp_) << file_name << " could not be opened.";
}

InputAudioFile::~InputAudioFile() {
  RTC_DCHECK(fp_);
  fclose(fp_);
}

bool InputAudioFile::Read(size_t samples, int16_t* destination) {
  if (!fp_) {
    return false;
  }
  size_t samples_read = fread(destination, sizeof(int16_t), samples, fp_);
  if (samples_read < samples) {
    if (!loop_at_end_) {
      return false;
    }
    // Rewind and read the missing samples.
    rewind(fp_);
    size_t missing_samples = samples - samples_read;
    if (fread(destination + samples_read, sizeof(int16_t), missing_samples,
              fp_) < missing_samples) {
      // Could not read enough even after rewinding the file.
      return false;
    }
  }
  return true;
}

bool InputAudioFile::Seek(int samples) {
  if (!fp_) {
    return false;
  }
  // Find file boundaries.
  const long current_pos = ftell(fp_);
  RTC_CHECK_NE(EOF, current_pos)
      << "Error returned when getting file position.";
  RTC_CHECK_EQ(0, fseek(fp_, 0, SEEK_END));  // Move to end of file.
  const long file_size = ftell(fp_);
  RTC_CHECK_NE(EOF, file_size) << "Error returned when getting file position.";
  // Find new position.
  long new_pos = current_pos + sizeof(int16_t) * samples;  // Samples to bytes.
  if (loop_at_end_) {
    new_pos = new_pos % file_size;  // Wrap around the end of the file.
    if (new_pos < 0) {
      // For negative values of new_pos, newpos % file_size will also be
      // negative. To get the correct result it's needed to add file_size.
      new_pos += file_size;
    }
  } else {
    new_pos = new_pos > file_size ? file_size : new_pos;  // Don't loop.
  }
  RTC_CHECK_GE(new_pos, 0)
      << "Trying to move to before the beginning of the file";
  // Move to new position relative to the beginning of the file.
  RTC_CHECK_EQ(0, fseek(fp_, new_pos, SEEK_SET));
  return true;
}

void InputAudioFile::DuplicateInterleaved(const int16_t* source,
                                          size_t samples,
                                          size_t channels,
                                          int16_t* destination) {
  // Start from the end of `source` and `destination`, and work towards the
  // beginning. This is to allow in-place interleaving of the same array (i.e.,
  // `source` and `destination` are the same array).
  for (int i = static_cast<int>(samples - 1); i >= 0; --i) {
    for (int j = static_cast<int>(channels - 1); j >= 0; --j) {
      destination[i * channels + j] = source[i];
    }
  }
}

}  // namespace test
}  // namespace webrtc
