/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#include "video/corruption_detection/halton_sequence.h"

#include <algorithm>
#include <vector>

#include "rtc_base/checks.h"

namespace webrtc {
namespace {

static constexpr int kMaxDimensions = 5;
const int kBases[kMaxDimensions] = {2, 3, 5, 7, 11};

double GetVanDerCorputSequenceElement(int sequence_idx, int base) {
  if (sequence_idx < 0 || base < 2) {
    sequence_idx = 0;
    base = 2;
  }
  double element = 0.0;
  double positional_value = 1.0;
  int left = sequence_idx;
  while (left > 0) {
    positional_value /= base;
    element += positional_value * (left % base);
    left /= base;
  }
  return element;
}

}  // namespace

HaltonSequence::HaltonSequence(int num_dimensions)
    : num_dimensions_(num_dimensions), current_idx_(0) {
  RTC_CHECK_GE(num_dimensions_, 1)
      << "num_dimensions must be >= 1. Will be set to 1.";
  RTC_CHECK_LE(num_dimensions_, kMaxDimensions)
      << "num_dimensions must be <= " << kMaxDimensions << ". Will be set to "
      << kMaxDimensions << ".";
  num_dimensions_ = std::clamp(num_dimensions_, 1, kMaxDimensions);
}

std::vector<double> HaltonSequence::GetNext() {
  std::vector<double> point = {};
  point.reserve(num_dimensions_);
  for (int i = 0; i < num_dimensions_; ++i) {
    point.push_back(GetVanDerCorputSequenceElement(current_idx_, kBases[i]));
  }
  ++current_idx_;
  return point;
}

void HaltonSequence::SetCurrentIndex(int idx) {
  if (idx >= 0) {
    current_idx_ = idx;
  }
  RTC_DCHECK_GE(idx, 0) << "Index must be non-negative";
}

void HaltonSequence::Reset() {
  HaltonSequence::current_idx_ = 0;
}

}  // namespace webrtc
