/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#include "common/video_frame.h"

#include <cstdio>

namespace libwebm {

bool VideoFrame::Buffer::Init(std::size_t new_length) {
  capacity = 0;
  length = 0;
  data.reset(new std::uint8_t[new_length]);

  if (data.get() == nullptr) {
    fprintf(stderr, "VideoFrame: Out of memory.");
    return false;
  }

  capacity = new_length;
  length = 0;
  return true;
}

bool VideoFrame::Init(std::size_t length) { return buffer_.Init(length); }

bool VideoFrame::Init(std::size_t length, std::int64_t nano_pts, Codec codec) {
  nanosecond_pts_ = nano_pts;
  codec_ = codec;
  return Init(length);
}

bool VideoFrame::SetBufferLength(std::size_t length) {
  if (length > buffer_.capacity || buffer_.data.get() == nullptr)
    return false;

  buffer_.length = length;
  return true;
}

}  // namespace libwebm
