/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#ifndef MODULES_CONGESTION_CONTROLLER_REMB_THROTTLER_H_
#define MODULES_CONGESTION_CONTROLLER_REMB_THROTTLER_H_

#include <functional>
#include <vector>

#include "api/units/data_rate.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "modules/remote_bitrate_estimator/include/remote_bitrate_estimator.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// RembThrottler is a helper class used for throttling RTCP REMB messages.
// Throttles small changes to the received BWE within 200ms.
class RembThrottler : public RemoteBitrateObserver {
 public:
  using RembSender =
      std::function<void(int64_t bitrate_bps, std::vector<uint32_t> ssrcs)>;
  RembThrottler(RembSender remb_sender, Clock* clock);

  // Ensures the remote party is notified of the receive bitrate no larger than
  // `bitrate` using RTCP REMB.
  void SetMaxDesiredReceiveBitrate(DataRate bitrate);

  // Implements RemoteBitrateObserver;
  // Called every time there is a new bitrate estimate for a receive channel
  // group. This call will trigger a new RTCP REMB packet if the bitrate
  // estimate has decreased or if no RTCP REMB packet has been sent for
  // a certain time interval.
  void OnReceiveBitrateChanged(const std::vector<uint32_t>& ssrcs,
                               uint32_t bitrate_bps) override;

 private:
  const RembSender remb_sender_;
  Clock* const clock_;
  mutable Mutex mutex_;
  Timestamp last_remb_time_ RTC_GUARDED_BY(mutex_);
  DataRate last_send_remb_bitrate_ RTC_GUARDED_BY(mutex_);
  DataRate max_remb_bitrate_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc
#endif  // MODULES_CONGESTION_CONTROLLER_REMB_THROTTLER_H_
