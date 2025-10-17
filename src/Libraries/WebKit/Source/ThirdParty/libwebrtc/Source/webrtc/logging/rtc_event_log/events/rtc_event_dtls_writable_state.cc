/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
#include "logging/rtc_event_log/events/rtc_event_dtls_writable_state.h"

#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"

namespace webrtc {

RtcEventDtlsWritableState::RtcEventDtlsWritableState(bool writable)
    : writable_(writable) {}

RtcEventDtlsWritableState::RtcEventDtlsWritableState(
    const RtcEventDtlsWritableState& other)
    : RtcEvent(other.timestamp_us_), writable_(other.writable_) {}

RtcEventDtlsWritableState::~RtcEventDtlsWritableState() = default;

std::unique_ptr<RtcEventDtlsWritableState> RtcEventDtlsWritableState::Copy()
    const {
  return absl::WrapUnique<RtcEventDtlsWritableState>(
      new RtcEventDtlsWritableState(*this));
}

}  // namespace webrtc
