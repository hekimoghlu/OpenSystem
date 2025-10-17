/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
#include "api/video/video_bitrate_allocator.h"

#include <cstdint>

#include "api/units/data_rate.h"
#include "api/video/video_bitrate_allocation.h"

namespace webrtc {

VideoBitrateAllocationParameters::VideoBitrateAllocationParameters(
    uint32_t total_bitrate_bps,
    uint32_t framerate)
    : total_bitrate(DataRate::BitsPerSec(total_bitrate_bps)),
      stable_bitrate(DataRate::BitsPerSec(total_bitrate_bps)),
      framerate(static_cast<double>(framerate)) {}

VideoBitrateAllocationParameters::VideoBitrateAllocationParameters(
    DataRate total_bitrate,
    double framerate)
    : total_bitrate(total_bitrate),
      stable_bitrate(total_bitrate),
      framerate(framerate) {}

VideoBitrateAllocationParameters::VideoBitrateAllocationParameters(
    DataRate total_bitrate,
    DataRate stable_bitrate,
    double framerate)
    : total_bitrate(total_bitrate),
      stable_bitrate(stable_bitrate),
      framerate(framerate) {}

VideoBitrateAllocationParameters::~VideoBitrateAllocationParameters() = default;

VideoBitrateAllocation VideoBitrateAllocator::GetAllocation(
    uint32_t total_bitrate_bps,
    uint32_t framerate) {
  return Allocate({DataRate::BitsPerSec(total_bitrate_bps),
                   DataRate::BitsPerSec(total_bitrate_bps),
                   static_cast<double>(framerate)});
}

VideoBitrateAllocation VideoBitrateAllocator::Allocate(
    VideoBitrateAllocationParameters parameters) {
  return GetAllocation(parameters.total_bitrate.bps(), parameters.framerate);
}

void VideoBitrateAllocator::SetLegacyConferenceMode(bool /* enabled */) {}

}  // namespace webrtc
