/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#ifndef SRC_VIRTUAL_BLOCK_PARSER_H_
#define SRC_VIRTUAL_BLOCK_PARSER_H_

#include <cassert>
#include <cstdint>

#include "src/block_header_parser.h"
#include "src/element_parser.h"
#include "webm/callback.h"
#include "webm/dom_types.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#BlockVirtual
// http://www.webmproject.org/docs/container/#BlockVirtual
// http://matroska.org/technical/specs/index.html#block_virtual
class VirtualBlockParser : public ElementParser {
 public:
  Status Init(const ElementMetadata& metadata, std::uint64_t max_size) override;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  // Gets the parsed block header information. This must not be called until the
  // parse has been successfully completed.
  const VirtualBlock& value() const {
    assert(state_ == State::kDone);
    return value_;
  }

  // Gets the parsed block header information. This must not be called until the
  // parse has been successfully completed.
  VirtualBlock* mutable_value() {
    assert(state_ == State::kDone);
    return &value_;
  }

 private:
  std::uint64_t my_size_;
  std::uint64_t total_bytes_read_ = 0;

  VirtualBlock value_{};

  BlockHeaderParser parser_;

  enum class State {
    /* clang-format off */
    // State             Transitions to state  When
    kReadingHeader,   // kValidatingSize       header parsed
    kValidatingSize,  // kDone                 no errors
    kDone,            // No transitions from here (must call Init)
    /* clang-format on */
  } state_ = State::kReadingHeader;
};

}  // namespace webm

#endif  // SRC_VIRTUAL_BLOCK_PARSER_H_
