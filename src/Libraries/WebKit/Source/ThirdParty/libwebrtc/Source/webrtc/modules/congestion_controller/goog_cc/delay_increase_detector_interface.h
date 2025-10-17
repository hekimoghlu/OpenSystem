/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_DELAY_INCREASE_DETECTOR_INTERFACE_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_DELAY_INCREASE_DETECTOR_INTERFACE_H_

#include <stdint.h>

#include <cstddef>

#include "api/transport/bandwidth_usage.h"

namespace webrtc {

class DelayIncreaseDetectorInterface {
 public:
  DelayIncreaseDetectorInterface() {}
  virtual ~DelayIncreaseDetectorInterface() {}

  DelayIncreaseDetectorInterface(const DelayIncreaseDetectorInterface&) =
      delete;
  DelayIncreaseDetectorInterface& operator=(
      const DelayIncreaseDetectorInterface&) = delete;

  // Update the detector with a new sample. The deltas should represent deltas
  // between timestamp groups as defined by the InterArrival class.
  virtual void Update(double recv_delta_ms,
                      double send_delta_ms,
                      int64_t send_time_ms,
                      int64_t arrival_time_ms,
                      size_t packet_size,
                      bool calculated_deltas) = 0;

  virtual BandwidthUsage State() const = 0;
};

}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_DELAY_INCREASE_DETECTOR_INTERFACE_H_
