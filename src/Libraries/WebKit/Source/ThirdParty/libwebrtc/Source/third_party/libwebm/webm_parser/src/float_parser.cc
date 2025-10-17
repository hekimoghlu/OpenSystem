/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/float_parser.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>

#include "src/parser_utils.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

FloatParser::FloatParser(double default_value)
    : default_value_(default_value) {}

Status FloatParser::Init(const ElementMetadata& metadata,
                         std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  if (metadata.size == 0) {
    value_ = default_value_;
  } else if (metadata.size == 4 || metadata.size == 8) {
    uint64_value_ = 0;
  } else {
    return Status(Status::kInvalidElementSize);
  }

  num_bytes_remaining_ = static_cast<int>(metadata.size);
  use_32_bits_ = metadata.size == 4;

  return Status(Status::kOkCompleted);
}

Status FloatParser::Feed(Callback* callback, Reader* reader,
                         std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  if (num_bytes_remaining_ == 0) {
    return Status(Status::kOkCompleted);
  }

  const Status status = AccumulateIntegerBytes(num_bytes_remaining_, reader,
                                               &uint64_value_, num_bytes_read);
  num_bytes_remaining_ -= static_cast<int>(*num_bytes_read);

  if (num_bytes_remaining_ == 0) {
    if (use_32_bits_) {
      static_assert(std::numeric_limits<float>::is_iec559,
                    "Your compiler does not support 32-bit IEC 559/IEEE 754 "
                    "floating point types");
      std::uint32_t uint32_value = static_cast<std::uint32_t>(uint64_value_);
      float float32_value;
      std::memcpy(&float32_value, &uint32_value, 4);
      value_ = float32_value;
    } else {
      static_assert(std::numeric_limits<double>::is_iec559,
                    "Your compiler does not support 64-bit IEC 559/IEEE 754 "
                    "floating point types");
      std::memcpy(&value_, &uint64_value_, 8);
    }
  }

  return status;
}

}  // namespace webm
