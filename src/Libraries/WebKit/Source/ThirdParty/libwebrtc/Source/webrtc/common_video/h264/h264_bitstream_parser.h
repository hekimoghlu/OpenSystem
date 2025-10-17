/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
#ifndef COMMON_VIDEO_H264_H264_BITSTREAM_PARSER_H_
#define COMMON_VIDEO_H264_H264_BITSTREAM_PARSER_H_
#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/video_codecs/bitstream_parser.h"
#include "common_video/h264/pps_parser.h"
#include "common_video/h264/sps_parser.h"

namespace webrtc {

// Stateful H264 bitstream parser (due to SPS/PPS). Used to parse out QP values
// from the bitstream.
// TODO(pbos): Unify with RTP SPS parsing and only use one H264 parser.
// TODO(pbos): If/when this gets used on the receiver side CHECKs must be
// removed and gracefully abort as we have no control over receive-side
// bitstreams.
class H264BitstreamParser : public BitstreamParser {
 public:
  H264BitstreamParser();
  ~H264BitstreamParser() override;

  void ParseBitstream(rtc::ArrayView<const uint8_t> bitstream) override;
  std::optional<int> GetLastSliceQp() const override;

 protected:
  enum Result {
    kOk,
    kInvalidStream,
    kUnsupportedStream,
  };
  void ParseSlice(rtc::ArrayView<const uint8_t> slice);
  Result ParseNonParameterSetNalu(rtc::ArrayView<const uint8_t> source,
                                  uint8_t nalu_type);

  // SPS/PPS state, updated when parsing new SPS/PPS, used to parse slices.
  std::optional<SpsParser::SpsState> sps_;
  std::optional<PpsParser::PpsState> pps_;

  // Last parsed slice QP.
  std::optional<int32_t> last_slice_qp_delta_;
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_H264_H264_BITSTREAM_PARSER_H_
