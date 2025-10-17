/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#ifndef API_VIDEO_VIDEO_BITRATE_ALLOCATOR_H_
#define API_VIDEO_VIDEO_BITRATE_ALLOCATOR_H_

#include <cstdint>

#include "api/units/data_rate.h"
#include "api/video/video_bitrate_allocation.h"

namespace webrtc {

struct VideoBitrateAllocationParameters {
  VideoBitrateAllocationParameters(uint32_t total_bitrate_bps,
                                   uint32_t framerate);
  VideoBitrateAllocationParameters(DataRate total_bitrate, double framerate);
  VideoBitrateAllocationParameters(DataRate total_bitrate,
                                   DataRate stable_bitrate,
                                   double framerate);
  ~VideoBitrateAllocationParameters();

  DataRate total_bitrate;
  DataRate stable_bitrate;
  double framerate;
};

class VideoBitrateAllocator {
 public:
  VideoBitrateAllocator() {}
  virtual ~VideoBitrateAllocator() {}

  virtual VideoBitrateAllocation GetAllocation(uint32_t total_bitrate_bps,
                                               uint32_t framerate);

  virtual VideoBitrateAllocation Allocate(
      VideoBitrateAllocationParameters parameters);

  // Deprecated: Only used to work around issues with the legacy conference
  // screenshare mode and shouldn't be needed by any subclasses.
  virtual void SetLegacyConferenceMode(bool enabled);
};

class VideoBitrateAllocationObserver {
 public:
  VideoBitrateAllocationObserver() {}
  virtual ~VideoBitrateAllocationObserver() {}

  virtual void OnBitrateAllocationUpdated(
      const VideoBitrateAllocation& allocation) = 0;
};

}  // namespace webrtc

#endif  // API_VIDEO_VIDEO_BITRATE_ALLOCATOR_H_
