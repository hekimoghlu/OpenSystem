/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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
#include <math.h>
#include <string.h>

#include <algorithm>
#include <bitset>
#include <vector>

#include "api/audio/echo_detector_creator.h"
#include "rtc_base/checks.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  // Number of times to update the echo detector.
  constexpr size_t kNrOfUpdates = 7;
  // Each round of updates requires a call to both AnalyzeRender and
  // AnalyzeCapture, so the amount of needed input bytes doubles. Also, two
  // bytes are used to set the call order.
  constexpr size_t kNrOfNeededInputBytes = 2 * kNrOfUpdates * sizeof(float) + 2;
  // The maximum audio energy that an audio frame can have is equal to the
  // number of samples in the frame multiplied by 2^30. We use a single sample
  // to represent an audio frame in this test, so it should have a maximum value
  // equal to the square root of that value.
  const float maxFuzzedValue = sqrtf(20 * 48) * 32768;
  if (size < kNrOfNeededInputBytes) {
    return;
  }
  size_t read_idx = 0;
  // Use the first two bytes to choose the call order.
  uint16_t call_order_int;
  memcpy(&call_order_int, &data[read_idx], 2);
  read_idx += 2;
  std::bitset<16> call_order(call_order_int);

  rtc::scoped_refptr<EchoDetector> echo_detector = CreateEchoDetector();
  std::vector<float> input(1);
  // Call AnalyzeCaptureAudio once to prevent the flushing of the buffer.
  echo_detector->AnalyzeCaptureAudio(input);
  for (size_t i = 0; i < 2 * kNrOfUpdates; ++i) {
    // Convert 4 input bytes to a float.
    RTC_DCHECK_LE(read_idx + sizeof(float), size);
    memcpy(input.data(), &data[read_idx], sizeof(float));
    read_idx += sizeof(float);
    if (!isfinite(input[0]) || fabs(input[0]) > maxFuzzedValue) {
      // Ignore infinity, nan values and values that are unrealistically large.
      continue;
    }
    if (call_order[i]) {
      echo_detector->AnalyzeRenderAudio(input);
    } else {
      echo_detector->AnalyzeCaptureAudio(input);
    }
  }
}

}  // namespace webrtc
