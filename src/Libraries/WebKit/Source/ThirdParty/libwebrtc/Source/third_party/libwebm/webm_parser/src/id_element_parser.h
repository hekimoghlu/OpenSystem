/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#ifndef SRC_ID_ELEMENT_PARSER_H_
#define SRC_ID_ELEMENT_PARSER_H_

#include <cstdint>

#include "src/element_parser.h"
#include "webm/callback.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

class IdElementParser : public ElementParser {
 public:
  IdElementParser() = default;

  IdElementParser(IdElementParser&&) = default;
  IdElementParser& operator=(IdElementParser&&) = default;

  IdElementParser(const IdElementParser&) = delete;
  IdElementParser& operator=(const IdElementParser&) = delete;

  Status Init(const ElementMetadata& metadata, std::uint64_t max_size) override;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  // Gets the parsed Id. This must not be called until the parse had been
  // successfully completed.
  Id value() const {
    assert(num_bytes_remaining_ == 0);
    return value_;
  }

  // Gets the parsed Id. This must not be called until the parse had been
  // successfully completed.
  Id* mutable_value() {
    assert(num_bytes_remaining_ == 0);
    return &value_;
  }

 private:
  Id value_;
  int num_bytes_remaining_ = -1;
};

}  // namespace webm

#endif  // SRC_ID_ELEMENT_PARSER_H_
