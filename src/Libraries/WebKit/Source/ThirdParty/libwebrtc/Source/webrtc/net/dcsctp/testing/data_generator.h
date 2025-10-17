/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#ifndef NET_DCSCTP_TESTING_DATA_GENERATOR_H_
#define NET_DCSCTP_TESTING_DATA_GENERATOR_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/data.h"

namespace dcsctp {

struct DataGeneratorOptions {
  StreamID stream_id = StreamID(1);
  std::optional<MID> mid = std::nullopt;
  PPID ppid = PPID(53);
};

// Generates Data with correct sequence numbers, and used only in unit tests.
class DataGenerator {
 public:
  explicit DataGenerator(MID start_mid = MID(0)) : mid_(start_mid) {}

  // Generates ordered "data" with the provided `payload` and flags, which can
  // contain "B" for setting the "is_beginning" flag, and/or "E" for setting the
  // "is_end" flag.
  Data Ordered(std::vector<uint8_t> payload,
               absl::string_view flags = "",
               DataGeneratorOptions opts = {});

  // Generates unordered "data" with the provided `payload` and flags, which can
  // contain "B" for setting the "is_beginning" flag, and/or "E" for setting the
  // "is_end" flag.
  Data Unordered(std::vector<uint8_t> payload,
                 absl::string_view flags = "",
                 DataGeneratorOptions opts = {});

  // Resets the Message ID identifier - simulating a "stream reset".
  void ResetStream() { mid_ = MID(0); }

 private:
  MID mid_;
  FSN fsn_ = FSN(0);
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_TESTING_DATA_GENERATOR_H_
