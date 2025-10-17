/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#ifndef LOGGING_RTC_EVENT_LOG_FAKE_RTC_EVENT_LOG_FACTORY_H_
#define LOGGING_RTC_EVENT_LOG_FAKE_RTC_EVENT_LOG_FACTORY_H_

#include <memory>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/rtc_event_log/rtc_event_log_factory_interface.h"
#include "logging/rtc_event_log/fake_rtc_event_log.h"

namespace webrtc {

class FakeRtcEventLogFactory : public RtcEventLogFactoryInterface {
 public:
  FakeRtcEventLogFactory() = default;
  ~FakeRtcEventLogFactory() override = default;

  absl::Nonnull<std::unique_ptr<RtcEventLog>> Create(
      const Environment& env) const override;

  FakeRtcEventLog* last_log_created() { return last_log_created_; }

 private:
  FakeRtcEventLog* last_log_created_ = nullptr;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_FAKE_RTC_EVENT_LOG_FACTORY_H_
