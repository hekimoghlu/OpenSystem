/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#include "src/block_header_parser.h"

#include <cassert>
#include <cstdint>

#include "src/parser_utils.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#block_structure
// http://matroska.org/technical/specs/index.html#simpleblock_structure
// http://matroska.org/technical/specs/index.html#block_virtual
Status BlockHeaderParser::Feed(Callback* callback, Reader* reader,
                               std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  *num_bytes_read = 0;

  Status status;
  std::uint64_t local_num_bytes_read;

  while (true) {
    switch (state_) {
      case State::kReadingTrackNumber: {
        status = uint_parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        if (!status.completed_ok()) {
          return status;
        }
        value_.track_number = uint_parser_.value();
        state_ = State::kReadingTimecode;
        continue;
      }

      case State::kReadingTimecode: {
        status =
            AccumulateIntegerBytes(timecode_bytes_remaining_, reader,
                                   &value_.timecode, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        timecode_bytes_remaining_ -= static_cast<int>(local_num_bytes_read);
        if (!status.completed_ok()) {
          return status;
        }
        state_ = State::kReadingFlags;
        continue;
      }

      case State::kReadingFlags: {
        assert(timecode_bytes_remaining_ == 0);
        status = ReadByte(reader, &value_.flags);
        if (!status.completed_ok()) {
          return status;
        }
        ++*num_bytes_read;
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
