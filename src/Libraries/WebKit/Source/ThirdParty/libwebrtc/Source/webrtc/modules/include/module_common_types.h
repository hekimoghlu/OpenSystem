/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#ifndef MODULES_INCLUDE_MODULE_COMMON_TYPES_H_
#define MODULES_INCLUDE_MODULE_COMMON_TYPES_H_

#include <stdint.h>

#include <vector>

namespace webrtc {

// Interface used by the CallStats class to distribute call statistics.
// Callbacks will be triggered as soon as the class has been registered to a
// CallStats object using RegisterStatsObserver.
class CallStatsObserver {
 public:
  virtual void OnRttUpdate(int64_t avg_rtt_ms, int64_t max_rtt_ms) = 0;

  virtual ~CallStatsObserver() {}
};

// Interface used by NackModule and JitterBuffer.
class NackSender {
 public:
  // If `buffering_allowed`, other feedback messages (e.g. key frame requests)
  // may be added to the same outgoing feedback message. In that case, it's up
  // to the user of the interface to ensure that when all buffer-able messages
  // have been added, the feedback message is triggered.
  virtual void SendNack(const std::vector<uint16_t>& sequence_numbers,
                        bool buffering_allowed) = 0;

 protected:
  virtual ~NackSender() {}
};

// Interface used by NackModule and JitterBuffer.
class KeyFrameRequestSender {
 public:
  virtual void RequestKeyFrame() = 0;

 protected:
  virtual ~KeyFrameRequestSender() {}
};

// Interface used by LossNotificationController to communicate to RtpRtcp.
class LossNotificationSender {
 public:
  virtual ~LossNotificationSender() {}

  virtual void SendLossNotification(uint16_t last_decoded_seq_num,
                                    uint16_t last_received_seq_num,
                                    bool decodability_flag,
                                    bool buffering_allowed) = 0;
};

}  // namespace webrtc

#endif  // MODULES_INCLUDE_MODULE_COMMON_TYPES_H_
