/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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
#ifndef CALL_RTP_TRANSPORT_CONTROLLER_SEND_FACTORY_H_
#define CALL_RTP_TRANSPORT_CONTROLLER_SEND_FACTORY_H_

#include <memory>

#include "call/rtp_transport_config.h"
#include "call/rtp_transport_controller_send.h"
#include "call/rtp_transport_controller_send_factory_interface.h"
#include "call/rtp_transport_controller_send_interface.h"

namespace webrtc {
class RtpTransportControllerSendFactory
    : public RtpTransportControllerSendFactoryInterface {
 public:
  std::unique_ptr<RtpTransportControllerSendInterface> Create(
      const RtpTransportConfig& config) override {
    return std::make_unique<RtpTransportControllerSend>(config);
  }

  virtual ~RtpTransportControllerSendFactory() {}
};
}  // namespace webrtc
#endif  // CALL_RTP_TRANSPORT_CONTROLLER_SEND_FACTORY_H_
