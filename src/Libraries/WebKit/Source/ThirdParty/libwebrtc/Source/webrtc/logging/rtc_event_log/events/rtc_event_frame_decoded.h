/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_FRAME_DECODED_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_FRAME_DECODED_H_

#include <stdint.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "api/video/video_codec_type.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

struct LoggedFrameDecoded {
  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  int64_t render_time_ms;
  uint32_t ssrc;
  int width;
  int height;
  VideoCodecType codec;
  uint8_t qp;
};

class RtcEventFrameDecoded final : public RtcEvent {
 public:
  static constexpr Type kType = Type::FrameDecoded;

  RtcEventFrameDecoded(int64_t render_time_ms,
                       uint32_t ssrc,
                       int width,
                       int height,
                       VideoCodecType codec,
                       uint8_t qp);
  ~RtcEventFrameDecoded() override = default;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  std::unique_ptr<RtcEventFrameDecoded> Copy() const;

  int64_t render_time_ms() const { return render_time_ms_; }
  uint32_t ssrc() const { return ssrc_; }
  int width() const { return width_; }
  int height() const { return height_; }
  VideoCodecType codec() const { return codec_; }
  uint8_t qp() const { return qp_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::map<uint32_t, std::vector<LoggedFrameDecoded>>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventFrameDecoded(const RtcEventFrameDecoded& other);

  const int64_t render_time_ms_;
  const uint32_t ssrc_;
  const int width_;
  const int height_;
  const VideoCodecType codec_;
  const uint8_t qp_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_FRAME_DECODED_H_
