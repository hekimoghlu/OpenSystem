/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_SPEECH_PROBABILITY_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_SPEECH_PROBABILITY_BUFFER_H_

#include <vector>

#include "rtc_base/gtest_prod_util.h"

namespace webrtc {

// This class implements a circular buffer that stores speech probabilities
// for a speech segment and estimates speech activity for that segment.
class SpeechProbabilityBuffer {
 public:
  // Ctor. The value of `low_probability_threshold` is required to be on the
  // range [0.0f, 1.0f].
  explicit SpeechProbabilityBuffer(float low_probability_threshold);
  ~SpeechProbabilityBuffer() {}
  SpeechProbabilityBuffer(const SpeechProbabilityBuffer&) = delete;
  SpeechProbabilityBuffer& operator=(const SpeechProbabilityBuffer&) = delete;

  // Adds `probability` in the buffer and computes an updatds sum of the buffer
  // probabilities. Value of `probability` is required to be on the range
  // [0.0f, 1.0f].
  void Update(float probability);

  // Resets the histogram, forgets the past.
  void Reset();

  // Returns true if the segment is active (a long enough segment with an
  // average speech probability above `low_probability_threshold`).
  bool IsActiveSegment() const;

 private:
  void RemoveTransient();

  // Use only for testing.
  float GetSumProbabilities() const { return sum_probabilities_; }

  FRIEND_TEST_ALL_PREFIXES(SpeechProbabilityBufferTest,
                           CheckSumAfterInitialization);
  FRIEND_TEST_ALL_PREFIXES(SpeechProbabilityBufferTest, CheckSumAfterUpdate);
  FRIEND_TEST_ALL_PREFIXES(SpeechProbabilityBufferTest, CheckSumAfterReset);
  FRIEND_TEST_ALL_PREFIXES(SpeechProbabilityBufferTest,
                           CheckSumAfterTransientNotRemoved);
  FRIEND_TEST_ALL_PREFIXES(SpeechProbabilityBufferTest,
                           CheckSumAfterTransientRemoved);

  const float low_probability_threshold_;

  // Sum of probabilities stored in `probabilities_`. Must be updated if
  // `probabilities_` is updated.
  float sum_probabilities_ = 0.0f;

  // Circular buffer for probabilities.
  std::vector<float> probabilities_;

  // Current index of the circular buffer, where the newest data will be written
  // to, therefore, pointing to the oldest data if buffer is full.
  int buffer_index_ = 0;

  // Indicates if the buffer is full and adding a new value removes the oldest
  // value.
  int buffer_is_full_ = false;

  int num_high_probability_observations_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_SPEECH_PROBABILITY_BUFFER_H_
