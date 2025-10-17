/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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
#ifndef API_RTC_EVENT_LOG_RTC_EVENT_LOG_H_
#define API_RTC_EVENT_LOG_RTC_EVENT_LOG_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "api/rtc_event_log/rtc_event.h"
#include "api/rtc_event_log_output.h"

namespace webrtc {

class RtcEventLog {
 public:
  enum : size_t { kUnlimitedOutput = 0 };
  enum : int64_t { kImmediateOutput = 0 };

  // TODO(eladalon):  Get rid of the legacy encoding and this enum once all
  // clients have migrated to the new format.
  enum class EncodingType { Legacy, NewFormat, ProtoFree };

  virtual ~RtcEventLog() = default;

  // Starts logging to a given output. The output might be limited in size,
  // and may close itself once it has reached the maximum size.
  virtual bool StartLogging(std::unique_ptr<RtcEventLogOutput> output,
                            int64_t output_period_ms) = 0;

  // Stops logging to file and waits until the file has been closed, after
  // which it would be permissible to read and/or modify it.
  virtual void StopLogging() = 0;

  // Stops logging to file and calls `callback` when the file has been closed.
  // Note that it is not safe to call any other members, including the
  // destructor, until the callback has been called.
  // TODO(srte): Remove default implementation when it's safe to do so.
  virtual void StopLogging(std::function<void()> callback) {
    StopLogging();
    callback();
  }

  // Log an RTC event (the type of event is determined by the subclass).
  virtual void Log(std::unique_ptr<RtcEvent> event) = 0;
};

// No-op implementation is used if flag is not set, or in tests.
class RtcEventLogNull final : public RtcEventLog {
 public:
  bool StartLogging(std::unique_ptr<RtcEventLogOutput> output,
                    int64_t output_period_ms) override;
  void StopLogging() override {}
  void Log(std::unique_ptr<RtcEvent> /* event */) override {}
};

}  // namespace webrtc

#endif  // API_RTC_EVENT_LOG_RTC_EVENT_LOG_H_
