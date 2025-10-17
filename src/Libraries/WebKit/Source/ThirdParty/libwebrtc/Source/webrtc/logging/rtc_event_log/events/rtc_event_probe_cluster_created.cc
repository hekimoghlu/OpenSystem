/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
#include "logging/rtc_event_log/events/rtc_event_probe_cluster_created.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"

namespace webrtc {

RtcEventProbeClusterCreated::RtcEventProbeClusterCreated(int32_t id,
                                                         int32_t bitrate_bps,
                                                         uint32_t min_probes,
                                                         uint32_t min_bytes)
    : id_(id),
      bitrate_bps_(bitrate_bps),
      min_probes_(min_probes),
      min_bytes_(min_bytes) {}

RtcEventProbeClusterCreated::RtcEventProbeClusterCreated(
    const RtcEventProbeClusterCreated& other)
    : RtcEvent(other.timestamp_us_),
      id_(other.id_),
      bitrate_bps_(other.bitrate_bps_),
      min_probes_(other.min_probes_),
      min_bytes_(other.min_bytes_) {}

std::unique_ptr<RtcEventProbeClusterCreated> RtcEventProbeClusterCreated::Copy()
    const {
  return absl::WrapUnique<RtcEventProbeClusterCreated>(
      new RtcEventProbeClusterCreated(*this));
}

}  // namespace webrtc
