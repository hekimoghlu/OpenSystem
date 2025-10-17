/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#ifndef MODULES_CONGESTION_CONTROLLER_RTP_CONTROL_HANDLER_H_
#define MODULES_CONGESTION_CONTROLLER_RTP_CONTROL_HANDLER_H_

#include <stdint.h>

#include <optional>

#include "api/sequence_checker.h"
#include "api/transport/network_types.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "rtc_base/system/no_unique_address.h"

namespace webrtc {
// This is used to observe the network controller state and route calls to
// the proper handler. It also keeps cached values for safe asynchronous use.
// This makes sure that things running on the worker queue can't access state
// in RtpTransportControllerSend, which would risk causing data race on
// destruction unless members are properly ordered.
class CongestionControlHandler {
 public:
  CongestionControlHandler() = default;

  CongestionControlHandler(const CongestionControlHandler&) = delete;
  CongestionControlHandler& operator=(const CongestionControlHandler&) = delete;

  ~CongestionControlHandler() = default;

  void SetTargetRate(TargetTransferRate new_target_rate);
  void SetNetworkAvailability(bool network_available);
  void SetPacerQueue(TimeDelta expected_queue_time);
  std::optional<TargetTransferRate> GetUpdate();

 private:
  std::optional<TargetTransferRate> last_incoming_;
  std::optional<TargetTransferRate> last_reported_;
  bool network_available_ = true;
  bool encoder_paused_in_last_report_ = false;

  int64_t pacer_expected_queue_ms_ = 0;

  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequenced_checker_;
};
}  // namespace webrtc
#endif  // MODULES_CONGESTION_CONTROLLER_RTP_CONTROL_HANDLER_H_
