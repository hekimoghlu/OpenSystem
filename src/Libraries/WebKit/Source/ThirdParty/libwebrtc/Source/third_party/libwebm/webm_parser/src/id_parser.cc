/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#include "src/id_parser.h"

#include <cassert>
#include <cstdint>

#include "src/bit_utils.h"
#include "src/parser_utils.h"
#include "webm/id.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

Status IdParser::Feed(Callback* callback, Reader* reader,
                      std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);
  assert(num_bytes_remaining_ != 0);

  *num_bytes_read = 0;

  // Spec references:
  // http://matroska.org/technical/specs/index.html#EBML_ex
  // https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#variable-size-integer
  // https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#element-id

  // IDs are encoded like so (big-endian):
  // 0b1xxx xxxx
  // 0b01xx xxxx  xxxx xxxx
  // 0b001x xxxx  xxxx xxxx  xxxx xxxx
  // 0b0001 xxxx  xxxx xxxx  xxxx xxxx  xxxx xxxx

  if (num_bytes_remaining_ == -1) {
    std::uint8_t first_byte;
    const Status status = ReadByte(reader, &first_byte);
    if (!status.completed_ok()) {
      return status;
    }
    ++*num_bytes_read;

    // The marker bit is the first 1-bit. It indicates the length of the ID.
    // If there is no marker bit in the first half-octet, this isn't a valid
    // ID, since IDs can't be more than 4 octets in MKV/WebM.
    if (!(first_byte & 0xf0)) {
      return Status(Status::kInvalidElementId);
    }

    num_bytes_remaining_ = CountLeadingZeros(first_byte);

    id_ = static_cast<Id>(first_byte);
  }

  std::uint64_t local_num_bytes_read;
  const Status status = AccumulateIntegerBytes(num_bytes_remaining_, reader,
                                               &id_, &local_num_bytes_read);
  *num_bytes_read += local_num_bytes_read;
  num_bytes_remaining_ -= static_cast<int>(local_num_bytes_read);

  return status;
}

Id IdParser::id() const {
  assert(num_bytes_remaining_ == 0);
  return id_;
}

}  // namespace webm
