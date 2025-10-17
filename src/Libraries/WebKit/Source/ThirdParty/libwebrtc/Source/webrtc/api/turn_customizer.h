/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#ifndef API_TURN_CUSTOMIZER_H_
#define API_TURN_CUSTOMIZER_H_

#include <stdlib.h>

#include "api/transport/stun.h"

namespace cricket {
class PortInterface;
}  // namespace cricket

namespace webrtc {

class TurnCustomizer {
 public:
  // This is called before a TURN message is sent.
  // This could be used to add implementation specific attributes to a request.
  virtual void MaybeModifyOutgoingStunMessage(
      cricket::PortInterface* port,
      cricket::StunMessage* message) = 0;

  // TURN can send data using channel data messages or Send indication.
  // This method should return false if `data` should be sent using
  // a Send indication instead of a ChannelData message, even if a
  // channel is bound.
  virtual bool AllowChannelData(cricket::PortInterface* port,
                                const void* data,
                                size_t size,
                                bool payload) = 0;

  virtual ~TurnCustomizer() {}
};

}  // namespace webrtc

#endif  // API_TURN_CUSTOMIZER_H_
