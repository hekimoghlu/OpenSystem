/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#include "modules/video_coding/utility/qp_parser.h"

#include "modules/video_coding/utility/vp8_header_parser.h"
#include "modules/video_coding/utility/vp9_uncompressed_header_parser.h"

namespace webrtc {

std::optional<uint32_t> QpParser::Parse(VideoCodecType codec_type,
                                        size_t spatial_idx,
                                        const uint8_t* frame_data,
                                        size_t frame_size) {
  if (frame_data == nullptr || frame_size == 0 ||
      spatial_idx >= kMaxSimulcastStreams) {
    return std::nullopt;
  }

  if (codec_type == kVideoCodecVP8) {
    int qp = -1;
    if (vp8::GetQp(frame_data, frame_size, &qp)) {
      return qp;
    }
  } else if (codec_type == kVideoCodecVP9) {
    int qp = -1;
    if (vp9::GetQp(frame_data, frame_size, &qp)) {
      return qp;
    }
  } else if (codec_type == kVideoCodecH264) {
    return h264_parsers_[spatial_idx].Parse(frame_data, frame_size);
  } else if (codec_type == kVideoCodecH265) {
    // H.265 bitstream parser is conditionally built.
#ifdef RTC_ENABLE_H265
    return h265_parsers_[spatial_idx].Parse(frame_data, frame_size);
#endif
  }

  return std::nullopt;
}

std::optional<uint32_t> QpParser::H264QpParser::Parse(const uint8_t* frame_data,
                                                      size_t frame_size) {
  MutexLock lock(&mutex_);
  bitstream_parser_.ParseBitstream(
      rtc::ArrayView<const uint8_t>(frame_data, frame_size));
  return bitstream_parser_.GetLastSliceQp();
}

#ifdef RTC_ENABLE_H265
std::optional<uint32_t> QpParser::H265QpParser::Parse(const uint8_t* frame_data,
                                                      size_t frame_size) {
  MutexLock lock(&mutex_);
  bitstream_parser_.ParseBitstream(
      rtc::ArrayView<const uint8_t>(frame_data, frame_size));
  return bitstream_parser_.GetLastSliceQp();
}
#endif

}  // namespace webrtc
