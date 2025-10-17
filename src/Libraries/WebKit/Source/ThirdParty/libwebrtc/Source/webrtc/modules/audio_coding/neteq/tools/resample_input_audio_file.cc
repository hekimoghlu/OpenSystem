/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include "modules/audio_coding/neteq/tools/resample_input_audio_file.h"

#include <memory>

#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

bool ResampleInputAudioFile::Read(size_t samples,
                                  int output_rate_hz,
                                  int16_t* destination) {
  const size_t samples_to_read = samples * file_rate_hz_ / output_rate_hz;
  RTC_CHECK_EQ(samples_to_read * output_rate_hz, samples * file_rate_hz_)
      << "Frame size and sample rates don't add up to an integer.";
  std::unique_ptr<int16_t[]> temp_destination(new int16_t[samples_to_read]);
  if (!InputAudioFile::Read(samples_to_read, temp_destination.get()))
    return false;
  resampler_.ResetIfNeeded(file_rate_hz_, output_rate_hz, 1);
  size_t output_length = 0;
  RTC_CHECK_EQ(resampler_.Push(temp_destination.get(), samples_to_read,
                               destination, samples, output_length),
               0);
  RTC_CHECK_EQ(samples, output_length);
  return true;
}

bool ResampleInputAudioFile::Read(size_t samples, int16_t* destination) {
  RTC_CHECK_GT(output_rate_hz_, 0) << "Output rate not set.";
  return Read(samples, output_rate_hz_, destination);
}

void ResampleInputAudioFile::set_output_rate_hz(int rate_hz) {
  output_rate_hz_ = rate_hz;
}

}  // namespace test
}  // namespace webrtc
