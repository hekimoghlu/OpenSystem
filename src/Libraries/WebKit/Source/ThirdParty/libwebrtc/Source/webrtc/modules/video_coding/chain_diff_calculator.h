/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#ifndef MODULES_VIDEO_CODING_CHAIN_DIFF_CALCULATOR_H_
#define MODULES_VIDEO_CODING_CHAIN_DIFF_CALCULATOR_H_

#include <stdint.h>

#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"

namespace webrtc {

// This class is thread compatible.
class ChainDiffCalculator {
 public:
  ChainDiffCalculator() = default;
  ChainDiffCalculator(const ChainDiffCalculator&) = default;
  ChainDiffCalculator& operator=(const ChainDiffCalculator&) = default;

  // Restarts chains, i.e. for position where chains[i] == true next chain_diff
  // will be 0. Saves chains.size() as number of chains in the stream.
  void Reset(const std::vector<bool>& chains);

  // Returns chain diffs based on flags if frame is part of the chain.
  absl::InlinedVector<int, 4> From(int64_t frame_id,
                                   const std::vector<bool>& chains);

 private:
  absl::InlinedVector<int, 4> ChainDiffs(int64_t frame_id) const;

  absl::InlinedVector<std::optional<int64_t>, 4> last_frame_in_chain_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CHAIN_DIFF_CALCULATOR_H_
