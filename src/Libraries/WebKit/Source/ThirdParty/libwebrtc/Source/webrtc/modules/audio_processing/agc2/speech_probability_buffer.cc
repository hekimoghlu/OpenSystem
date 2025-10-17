/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#include "modules/audio_processing/agc2/speech_probability_buffer.h"

#include <algorithm>

#include "rtc_base/checks.h"

namespace webrtc {
namespace {

constexpr float kActivityThreshold = 0.9f;
constexpr int kNumAnalysisFrames = 100;
// We use 12 in AGC2 adaptive digital, but with a slightly different logic.
constexpr int kTransientWidthThreshold = 7;

}  // namespace

SpeechProbabilityBuffer::SpeechProbabilityBuffer(
    float low_probability_threshold)
    : low_probability_threshold_(low_probability_threshold),
      probabilities_(kNumAnalysisFrames) {
  RTC_DCHECK_GE(low_probability_threshold, 0.0f);
  RTC_DCHECK_LE(low_probability_threshold, 1.0f);
  RTC_DCHECK(!probabilities_.empty());
}

void SpeechProbabilityBuffer::Update(float probability) {
  // Remove the oldest entry if the circular buffer is full.
  if (buffer_is_full_) {
    const float oldest_probability = probabilities_[buffer_index_];
    sum_probabilities_ -= oldest_probability;
  }

  // Check for transients.
  if (probability <= low_probability_threshold_) {
    // Set a probability lower than the threshold to zero.
    probability = 0.0f;

    // Check if this has been a transient.
    if (num_high_probability_observations_ <= kTransientWidthThreshold) {
      RemoveTransient();
    }
    num_high_probability_observations_ = 0;
  } else if (num_high_probability_observations_ <= kTransientWidthThreshold) {
    ++num_high_probability_observations_;
  }

  // Update the circular buffer and the current sum.
  probabilities_[buffer_index_] = probability;
  sum_probabilities_ += probability;

  // Increment the buffer index and check for wrap-around.
  if (++buffer_index_ >= kNumAnalysisFrames) {
    buffer_index_ = 0;
    buffer_is_full_ = true;
  }
}

void SpeechProbabilityBuffer::RemoveTransient() {
  // Don't expect to be here if high-activity region is longer than
  // `kTransientWidthThreshold` or there has not been any transient.
  RTC_DCHECK_LE(num_high_probability_observations_, kTransientWidthThreshold);

  // Replace previously added probabilities with zero.
  int index =
      (buffer_index_ > 0) ? (buffer_index_ - 1) : (kNumAnalysisFrames - 1);

  while (num_high_probability_observations_-- > 0) {
    sum_probabilities_ -= probabilities_[index];
    probabilities_[index] = 0.0f;

    // Update the circular buffer index.
    index = (index > 0) ? (index - 1) : (kNumAnalysisFrames - 1);
  }
}

bool SpeechProbabilityBuffer::IsActiveSegment() const {
  if (!buffer_is_full_) {
    return false;
  }
  if (sum_probabilities_ < kActivityThreshold * kNumAnalysisFrames) {
    return false;
  }
  return true;
}

void SpeechProbabilityBuffer::Reset() {
  sum_probabilities_ = 0.0f;

  // Empty the circular buffer.
  buffer_index_ = 0;
  buffer_is_full_ = false;
  num_high_probability_observations_ = 0;
}

}  // namespace webrtc
