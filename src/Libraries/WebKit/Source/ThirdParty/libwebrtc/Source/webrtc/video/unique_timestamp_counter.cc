/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#include "video/unique_timestamp_counter.h"

#include <cstdint>
#include <memory>
#include <set>

namespace webrtc {
namespace {

constexpr int kMaxHistory = 1000;

}  // namespace

UniqueTimestampCounter::UniqueTimestampCounter()
    : latest_(std::make_unique<uint32_t[]>(kMaxHistory)) {}

void UniqueTimestampCounter::Add(uint32_t value) {
  if (value == last_ || !search_index_.insert(value).second) {
    // Already known.
    return;
  }
  int index = unique_seen_ % kMaxHistory;
  if (unique_seen_ >= kMaxHistory) {
    search_index_.erase(latest_[index]);
  }
  latest_[index] = value;
  last_ = value;
  ++unique_seen_;
}

}  // namespace webrtc
