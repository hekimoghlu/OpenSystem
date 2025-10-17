/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
#include "api/rtc_event_log/rtc_event_log_factory.h"

#include <memory>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/field_trials_view.h"
#include "api/rtc_event_log/rtc_event_log.h"

#ifdef WEBRTC_ENABLE_RTC_EVENT_LOG
#include "logging/rtc_event_log/rtc_event_log_impl.h"
#endif

namespace webrtc {

absl::Nonnull<std::unique_ptr<RtcEventLog>> RtcEventLogFactory::Create(
    const Environment& env) const {
#ifndef WEBRTC_ENABLE_RTC_EVENT_LOG
  return std::make_unique<RtcEventLogNull>();
#else
  if (env.field_trials().IsEnabled("WebRTC-RtcEventLogKillSwitch")) {
    return std::make_unique<RtcEventLogNull>();
  }
  return std::make_unique<RtcEventLogImpl>(env);
#endif
}

}  // namespace webrtc
