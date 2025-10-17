/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_ENCODER_RTC_EVENT_LOG_ENCODER_V3_H_
#define LOGGING_RTC_EVENT_LOG_ENCODER_RTC_EVENT_LOG_ENCODER_V3_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "logging/rtc_event_log/encoder/rtc_event_log_encoder.h"

namespace webrtc {

class RtcEventLogEncoderV3 final : public RtcEventLogEncoder {
 public:
  RtcEventLogEncoderV3();
  ~RtcEventLogEncoderV3() override = default;

  std::string EncodeBatch(
      std::deque<std::unique_ptr<RtcEvent>>::const_iterator begin,
      std::deque<std::unique_ptr<RtcEvent>>::const_iterator end) override;

  std::string EncodeLogStart(int64_t timestamp_us,
                             int64_t utc_time_us) override;
  std::string EncodeLogEnd(int64_t timestamp_us) override;

 private:
  std::map<RtcEvent::Type,
           std::function<std::string(rtc::ArrayView<const RtcEvent*>)>>
      encoders_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_ENCODER_RTC_EVENT_LOG_ENCODER_V3_H_
