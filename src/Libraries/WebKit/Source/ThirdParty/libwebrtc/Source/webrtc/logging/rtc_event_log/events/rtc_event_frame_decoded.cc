/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_frame_decoded.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/video/video_codec_type.h"

namespace webrtc {

RtcEventFrameDecoded::RtcEventFrameDecoded(int64_t render_time_ms,
                                           uint32_t ssrc,
                                           int width,
                                           int height,
                                           VideoCodecType codec,
                                           uint8_t qp)
    : render_time_ms_(render_time_ms),
      ssrc_(ssrc),
      width_(width),
      height_(height),
      codec_(codec),
      qp_(qp) {}

RtcEventFrameDecoded::RtcEventFrameDecoded(const RtcEventFrameDecoded& other)
    : RtcEvent(other.timestamp_us_),
      render_time_ms_(other.render_time_ms_),
      ssrc_(other.ssrc_),
      width_(other.width_),
      height_(other.height_),
      codec_(other.codec_),
      qp_(other.qp_) {}

std::unique_ptr<RtcEventFrameDecoded> RtcEventFrameDecoded::Copy() const {
  return absl::WrapUnique<RtcEventFrameDecoded>(
      new RtcEventFrameDecoded(*this));
}

}  // namespace webrtc
