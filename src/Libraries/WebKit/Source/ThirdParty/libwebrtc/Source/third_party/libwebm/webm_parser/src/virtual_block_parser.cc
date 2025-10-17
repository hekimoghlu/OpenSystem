/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include "src/virtual_block_parser.h"

#include <cassert>
#include <cstdint>

#include "webm/element.h"

namespace webm {

Status VirtualBlockParser::Init(const ElementMetadata& metadata,
                                std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  if (metadata.size == kUnknownElementSize || metadata.size < 4) {
    return Status(Status::kInvalidElementSize);
  }

  *this = {};
  my_size_ = metadata.size;

  return Status(Status::kOkCompleted);
}

Status VirtualBlockParser::Feed(Callback* callback, Reader* reader,
                                std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  *num_bytes_read = 0;

  Status status;
  std::uint64_t local_num_bytes_read;

  while (true) {
    switch (state_) {
      case State::kReadingHeader: {
        status = parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        total_bytes_read_ += local_num_bytes_read;
        if (!status.completed_ok()) {
          return status;
        }
        value_.track_number = parser_.value().track_number;
        value_.timecode = parser_.value().timecode;
        state_ = State::kValidatingSize;
        continue;
      }

      case State::kValidatingSize: {
        if (my_size_ < total_bytes_read_) {
          return Status(Status::kInvalidElementValue);
        }
        state_ = State::kDone;
        continue;
      }

      case State::kDone: {
        return Status(Status::kOkCompleted);
      }
    }
  }
}

}  // namespace webm
