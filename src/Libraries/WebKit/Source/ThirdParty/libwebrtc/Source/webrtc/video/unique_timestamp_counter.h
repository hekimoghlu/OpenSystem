/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#ifndef VIDEO_UNIQUE_TIMESTAMP_COUNTER_H_
#define VIDEO_UNIQUE_TIMESTAMP_COUNTER_H_

#include <cstdint>
#include <memory>
#include <set>

namespace webrtc {

// Counts number of uniquely seen frames (aka pictures, aka temporal units)
// identified by their rtp timestamp.
class UniqueTimestampCounter {
 public:
  UniqueTimestampCounter();
  UniqueTimestampCounter(const UniqueTimestampCounter&) = delete;
  UniqueTimestampCounter& operator=(const UniqueTimestampCounter&) = delete;
  ~UniqueTimestampCounter() = default;

  void Add(uint32_t timestamp);
  // Returns number of different `timestamp` passed to the UniqueCounter.
  int GetUniqueSeen() const { return unique_seen_; }

 private:
  int unique_seen_ = 0;
  // Stores several last seen unique values for quick search.
  std::set<uint32_t> search_index_;
  // The same unique values in the circular buffer in the insertion order.
  std::unique_ptr<uint32_t[]> latest_;
  // Last inserted value for optimization purpose.
  int64_t last_ = -1;
};

}  // namespace webrtc

#endif  // VIDEO_UNIQUE_TIMESTAMP_COUNTER_H_
