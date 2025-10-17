/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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
#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include "webm/callback.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

class Parser {
 public:
  virtual ~Parser() = default;

  // Feeds data into the parser, with the number of bytes read from the reader
  // returned in num_bytes_read. Returns Status::kOkCompleted when parsing is
  // complete, or an appropriate error code if the data is malformed and cannot
  // be parsed. Otherwise, the status of Reader::Read is returned if only a
  // partial parse could be done because the reader couldn't immediately provide
  // all the needed data. reader and num_bytes_read must not be null. Do not
  // call again once the parse is complete.
  virtual Status Feed(Callback* callback, Reader* reader,
                      std::uint64_t* num_bytes_read) = 0;
};

}  // namespace webm

#endif  // SRC_PARSER_H_
