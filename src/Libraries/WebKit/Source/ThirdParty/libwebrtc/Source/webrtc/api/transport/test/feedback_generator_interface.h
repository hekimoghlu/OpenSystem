/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
#ifndef API_TRANSPORT_TEST_FEEDBACK_GENERATOR_INTERFACE_H_
#define API_TRANSPORT_TEST_FEEDBACK_GENERATOR_INTERFACE_H_

#include <cstddef>
#include <vector>

#include "api/test/simulated_network.h"
#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"

namespace webrtc {
class FeedbackGenerator {
 public:
  struct Config {
    BuiltInNetworkBehaviorConfig send_link;
    BuiltInNetworkBehaviorConfig return_link;
    TimeDelta feedback_interval = TimeDelta::Millis(50);
    DataSize feedback_packet_size = DataSize::Bytes(20);
  };
  virtual ~FeedbackGenerator() = default;
  virtual Timestamp Now() = 0;
  virtual void Sleep(TimeDelta duration) = 0;
  virtual void SendPacket(size_t size) = 0;
  virtual std::vector<TransportPacketsFeedback> PopFeedback() = 0;
  virtual void SetSendConfig(BuiltInNetworkBehaviorConfig config) = 0;
  virtual void SetReturnConfig(BuiltInNetworkBehaviorConfig config) = 0;
  virtual void SetSendLinkCapacity(DataRate capacity) = 0;
};
}  // namespace webrtc
#endif  // API_TRANSPORT_TEST_FEEDBACK_GENERATOR_INTERFACE_H_
