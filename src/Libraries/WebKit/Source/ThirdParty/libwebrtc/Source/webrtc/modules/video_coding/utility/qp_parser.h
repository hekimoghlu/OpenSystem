/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
#ifndef MODULES_VIDEO_CODING_UTILITY_QP_PARSER_H_
#define MODULES_VIDEO_CODING_UTILITY_QP_PARSER_H_

#include <optional>

#include "api/video/video_codec_constants.h"
#include "api/video/video_codec_type.h"
#include "common_video/h264/h264_bitstream_parser.h"
#ifdef RTC_ENABLE_H265
#include "common_video/h265/h265_bitstream_parser.h"
#endif
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
class QpParser {
 public:
  std::optional<uint32_t> Parse(VideoCodecType codec_type,
                                size_t spatial_idx,
                                const uint8_t* frame_data,
                                size_t frame_size);

 private:
  // A thread safe wrapper for H264 bitstream parser.
  class H264QpParser {
   public:
    std::optional<uint32_t> Parse(const uint8_t* frame_data, size_t frame_size);

   private:
    Mutex mutex_;
    H264BitstreamParser bitstream_parser_ RTC_GUARDED_BY(mutex_);
  };

  H264QpParser h264_parsers_[kMaxSimulcastStreams];

#ifdef RTC_ENABLE_H265
  // A thread safe wrapper for H.265 bitstream parser.
  class H265QpParser {
   public:
    std::optional<uint32_t> Parse(const uint8_t* frame_data, size_t frame_size);

   private:
    Mutex mutex_;
    H265BitstreamParser bitstream_parser_ RTC_GUARDED_BY(mutex_);
  };

  H265QpParser h265_parsers_[kMaxSimulcastStreams];
#endif
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_UTILITY_QP_PARSER_H_
