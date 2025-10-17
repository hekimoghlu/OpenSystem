/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#ifndef LOGGING_RTC_EVENT_LOG_FAKE_RTC_EVENT_LOG_H_
#define LOGGING_RTC_EVENT_LOG_FAKE_RTC_EVENT_LOG_H_

#include <cstdint>
#include <map>
#include <memory>

#include "api/rtc_event_log/rtc_event.h"
#include "api/rtc_event_log/rtc_event_log.h"
#include "api/rtc_event_log_output.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class FakeRtcEventLog : public RtcEventLog {
 public:
  FakeRtcEventLog() = default;
  ~FakeRtcEventLog() override = default;

  bool StartLogging(std::unique_ptr<RtcEventLogOutput> output,
                    int64_t output_period_ms) override;
  void StopLogging() override;
  void Log(std::unique_ptr<RtcEvent> event) override;
  int GetEventCount(RtcEvent::Type event_type);

 private:
  Mutex mu_;
  std::map<RtcEvent::Type, int> count_ RTC_GUARDED_BY(mu_);
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_FAKE_RTC_EVENT_LOG_H_
