/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
#ifndef API_VIDEO_CODECS_BITSTREAM_PARSER_H_
#define API_VIDEO_CODECS_BITSTREAM_PARSER_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/array_view.h"

namespace webrtc {

// This class is an interface for bitstream parsers.
class BitstreamParser {
 public:
  virtual ~BitstreamParser() = default;

  // Parse an additional chunk of the bitstream.
  virtual void ParseBitstream(rtc::ArrayView<const uint8_t> bitstream) = 0;

  // Get the last extracted QP value from the parsed bitstream. If no QP
  // value could be parsed, returns std::nullopt.
  virtual std::optional<int> GetLastSliceQp() const = 0;
};

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_BITSTREAM_PARSER_H_
