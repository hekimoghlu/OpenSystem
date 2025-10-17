/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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
#ifndef RTC_TOOLS_DATA_CHANNEL_BENCHMARK_SIGNALING_INTERFACE_H_
#define RTC_TOOLS_DATA_CHANNEL_BENCHMARK_SIGNALING_INTERFACE_H_

#include <memory>

#include "api/jsep.h"

namespace webrtc {
class SignalingInterface {
 public:
  virtual ~SignalingInterface() = default;

  // Send an ICE candidate over the transport.
  virtual void SendIceCandidate(
      const webrtc::IceCandidateInterface* candidate) = 0;

  // Send a local description over the transport.
  virtual void SendDescription(
      const webrtc::SessionDescriptionInterface* sdp) = 0;

  // Set a callback when receiving a description from the transport.
  virtual void OnRemoteDescription(
      std::function<void(std::unique_ptr<webrtc::SessionDescriptionInterface>
                             sdp)> callback) = 0;

  // Set a callback when receiving an ICE candidate from the transport.
  virtual void OnIceCandidate(
      std::function<void(std::unique_ptr<webrtc::IceCandidateInterface>
                             candidate)> callback) = 0;
};
}  // namespace webrtc

#endif  // RTC_TOOLS_DATA_CHANNEL_BENCHMARK_SIGNALING_INTERFACE_H_
