/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#ifndef COMMON_VIDEO_H264_SPS_VUI_REWRITER_H_
#define COMMON_VIDEO_H264_SPS_VUI_REWRITER_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/video/color_space.h"
#include "common_video/h264/sps_parser.h"
#include "rtc_base/buffer.h"

namespace webrtc {

// A class that can parse an SPS+VUI and if necessary creates a copy with
// updated parameters.
// The rewriter disables frame buffering. This should force decoders to deliver
// decoded frame immediately and, thus, reduce latency.
// The rewriter updates video signal type parameters if external parameters are
// provided.
class SpsVuiRewriter : private SpsParser {
 public:
  enum class ParseResult { kFailure, kVuiOk, kVuiRewritten };
  enum class Direction { kIncoming, kOutgoing };

  // Parses an SPS block and if necessary copies it and rewrites the VUI.
  // Returns kFailure on failure, kParseOk if parsing succeeded and no update
  // was necessary and kParsedAndModified if an updated copy of buffer was
  // written to destination. destination may be populated with some data even if
  // no rewrite was necessary, but the end offset should remain unchanged.
  // Unless parsing fails, the sps parameter will be populated with the parsed
  // SPS state. This function assumes that any previous headers
  // (NALU start, type, Stap-A, etc) have already been parsed and that RBSP
  // decoding has been performed.
  static ParseResult ParseAndRewriteSps(rtc::ArrayView<const uint8_t> buffer,
                                        std::optional<SpsParser::SpsState>* sps,
                                        const ColorSpace* color_space,
                                        rtc::Buffer* destination,
                                        Direction Direction);

  // Parses NAL units from `buffer`, strips AUD blocks and rewrites VUI in SPS
  // blocks if necessary.
  static rtc::Buffer ParseOutgoingBitstreamAndRewrite(
      rtc::ArrayView<const uint8_t> buffer,
      const ColorSpace* color_space);

 private:
  static ParseResult ParseAndRewriteSps(rtc::ArrayView<const uint8_t> buffer,
                                        std::optional<SpsParser::SpsState>* sps,
                                        const ColorSpace* color_space,
                                        rtc::Buffer* destination);

  static void UpdateStats(ParseResult result, Direction direction);
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_H264_SPS_VUI_REWRITER_H_
