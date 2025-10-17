/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#include "logging/rtc_event_log/fake_rtc_event_log_factory.h"

#include <memory>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/rtc_event_log/rtc_event_log.h"
#include "logging/rtc_event_log/fake_rtc_event_log.h"

namespace webrtc {

absl::Nonnull<std::unique_ptr<RtcEventLog>> FakeRtcEventLogFactory::Create(
    const Environment& /*env*/) const {
  auto fake_event_log = std::make_unique<FakeRtcEventLog>();
  const_cast<FakeRtcEventLog*&>(last_log_created_) = fake_event_log.get();
  return fake_event_log;
}

}  // namespace webrtc
