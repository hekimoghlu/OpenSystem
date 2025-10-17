/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "net/dcsctp/testing/data_generator.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "net/dcsctp/packet/data.h"
#include "net/dcsctp/public/types.h"

namespace dcsctp {
constexpr PPID kPpid = PPID(53);

Data DataGenerator::Ordered(std::vector<uint8_t> payload,
                            absl::string_view flags,
                            const DataGeneratorOptions opts) {
  Data::IsBeginning is_beginning(flags.find('B') != std::string::npos);
  Data::IsEnd is_end(flags.find('E') != std::string::npos);

  if (is_beginning) {
    fsn_ = FSN(0);
  } else {
    fsn_ = FSN(*fsn_ + 1);
  }
  MID mid = opts.mid.value_or(mid_);
  Data ret = Data(opts.stream_id, SSN(static_cast<uint16_t>(*mid)), mid, fsn_,
                  opts.ppid, std::move(payload), is_beginning, is_end,
                  IsUnordered(false));

  if (is_end) {
    mid_ = MID(*mid + 1);
  }
  return ret;
}

Data DataGenerator::Unordered(std::vector<uint8_t> payload,
                              absl::string_view flags,
                              const DataGeneratorOptions opts) {
  Data::IsBeginning is_beginning(flags.find('B') != std::string::npos);
  Data::IsEnd is_end(flags.find('E') != std::string::npos);

  if (is_beginning) {
    fsn_ = FSN(0);
  } else {
    fsn_ = FSN(*fsn_ + 1);
  }
  MID mid = opts.mid.value_or(mid_);
  Data ret = Data(opts.stream_id, SSN(0), mid, fsn_, kPpid, std::move(payload),
                  is_beginning, is_end, IsUnordered(true));
  if (is_end) {
    mid_ = MID(*mid + 1);
  }
  return ret;
}
}  // namespace dcsctp
