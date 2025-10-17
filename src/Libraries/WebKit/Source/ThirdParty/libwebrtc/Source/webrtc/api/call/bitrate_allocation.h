/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#ifndef API_CALL_BITRATE_ALLOCATION_H_
#define API_CALL_BITRATE_ALLOCATION_H_

#include "api/units/data_rate.h"
#include "api/units/time_delta.h"

namespace webrtc {

// BitrateAllocationUpdate provides information to allocated streams about their
// bitrate allocation. It originates from the BitrateAllocater class and is
// propagated from there.
struct BitrateAllocationUpdate {
  // The allocated target bitrate. Media streams should produce this amount of
  // data. (Note that this may include packet overhead depending on
  // configuration.)
  DataRate target_bitrate = DataRate::Zero();
  // The allocated part of the estimated link capacity. This is more stable than
  // the target as it is based on the underlying link capacity estimate. This
  // should be used to change encoder configuration when the cost of change is
  // high.
  DataRate stable_target_bitrate = DataRate::Zero();
  // Predicted packet loss ratio.
  double packet_loss_ratio = 0;
  // Predicted round trip time.
  TimeDelta round_trip_time = TimeDelta::PlusInfinity();
  // `bwe_period` is deprecated, use `stable_target_bitrate` allocation instead.
  TimeDelta bwe_period = TimeDelta::PlusInfinity();
  // Congestion window pushback bitrate reduction fraction. Used in
  // VideoStreamEncoder to reduce the bitrate by the given fraction
  // by dropping frames.
  double cwnd_reduce_ratio = 0;
};

}  // namespace webrtc

#endif  // API_CALL_BITRATE_ALLOCATION_H_
