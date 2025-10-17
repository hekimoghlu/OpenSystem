/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include "src/id_element_parser.h"

#include <cassert>
#include <cstdint>

#include "src/element_parser.h"
#include "src/parser_utils.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

Status IdElementParser::Init(const ElementMetadata& metadata,
                             std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  if (metadata.size == 0 || metadata.size > 4) {
    return Status(Status::kInvalidElementSize);
  }

  num_bytes_remaining_ = static_cast<int>(metadata.size);
  value_ = static_cast<Id>(0);

  return Status(Status::kOkCompleted);
}

Status IdElementParser::Feed(Callback* callback, Reader* reader,
                             std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  const Status status = AccumulateIntegerBytes(num_bytes_remaining_, reader,
                                               &value_, num_bytes_read);
  num_bytes_remaining_ -= static_cast<int>(*num_bytes_read);

  return status;
}

}  // namespace webm
