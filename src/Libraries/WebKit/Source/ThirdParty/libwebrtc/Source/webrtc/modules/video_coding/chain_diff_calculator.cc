/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
#include "modules/video_coding/chain_diff_calculator.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "rtc_base/logging.h"

namespace webrtc {

void ChainDiffCalculator::Reset(const std::vector<bool>& chains) {
  last_frame_in_chain_.resize(chains.size());
  for (size_t i = 0; i < chains.size(); ++i) {
    if (chains[i]) {
      last_frame_in_chain_[i] = std::nullopt;
    }
  }
}

absl::InlinedVector<int, 4> ChainDiffCalculator::ChainDiffs(
    int64_t frame_id) const {
  absl::InlinedVector<int, 4> result;
  result.reserve(last_frame_in_chain_.size());
  for (const auto& frame_id_in_chain : last_frame_in_chain_) {
    result.push_back(frame_id_in_chain ? (frame_id - *frame_id_in_chain) : 0);
  }
  return result;
}

absl::InlinedVector<int, 4> ChainDiffCalculator::From(
    int64_t frame_id,
    const std::vector<bool>& chains) {
  auto result = ChainDiffs(frame_id);
  if (chains.size() != last_frame_in_chain_.size()) {
    RTC_LOG(LS_ERROR) << "Insconsistent chain configuration for frame#"
                      << frame_id << ": expected "
                      << last_frame_in_chain_.size() << " chains, found "
                      << chains.size();
  }
  size_t num_chains = std::min(last_frame_in_chain_.size(), chains.size());
  for (size_t i = 0; i < num_chains; ++i) {
    if (chains[i]) {
      last_frame_in_chain_[i] = frame_id;
    }
  }
  return result;
}

}  // namespace webrtc
