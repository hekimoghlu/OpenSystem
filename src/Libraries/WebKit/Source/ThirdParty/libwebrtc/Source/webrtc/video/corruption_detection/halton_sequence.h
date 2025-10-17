/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#ifndef VIDEO_CORRUPTION_DETECTION_HALTON_SEQUENCE_H_
#define VIDEO_CORRUPTION_DETECTION_HALTON_SEQUENCE_H_

#include <vector>

namespace webrtc {

// Generates the Halton sequence: a low discrepancy sequence of doubles in the
// half-open interval [0,1). See https://en.wikipedia.org/wiki/Halton_sequence
// for information on how the sequence is constructed.
class HaltonSequence {
 public:
  // Creates a sequence in `num_dimensions` number of dimensions. Possible
  // values are [1, 5].
  explicit HaltonSequence(int num_dimensions);
  // Creates a default sequence in a single dimension.
  HaltonSequence() = default;
  HaltonSequence(const HaltonSequence&) = default;
  HaltonSequence(HaltonSequence&&) = default;
  HaltonSequence& operator=(const HaltonSequence&) = default;
  HaltonSequence& operator=(HaltonSequence&&) = default;
  ~HaltonSequence() = default;

  // Gets the next point in the sequence where each value is in the half-open
  // interval [0,1).
  std::vector<double> GetNext();
  int GetCurrentIndex() const { return current_idx_; }
  void SetCurrentIndex(int idx);
  void Reset();

 private:
  int num_dimensions_ = 1;
  int current_idx_ = 0;
};

}  // namespace webrtc

#endif  // VIDEO_CORRUPTION_DETECTION_HALTON_SEQUENCE_H_
