/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#ifndef API_TRANSPORT_SCTP_TRANSPORT_FACTORY_INTERFACE_H_
#define API_TRANSPORT_SCTP_TRANSPORT_FACTORY_INTERFACE_H_

#include <memory>

#include "api/environment/environment.h"

// These classes are not part of the API, and are treated as opaque pointers.
namespace cricket {
class SctpTransportInternal;
}  // namespace cricket

namespace rtc {
class PacketTransportInternal;
}  // namespace rtc

namespace webrtc {

// Factory class which can be used to allow fake SctpTransports to be injected
// for testing. An application is not intended to implement this interface nor
// 'cricket::SctpTransportInternal' because SctpTransportInternal is not
// guaranteed to remain stable in future WebRTC versions.
class SctpTransportFactoryInterface {
 public:
  virtual ~SctpTransportFactoryInterface() = default;

  // Create an SCTP transport using `channel` for the underlying transport.
  virtual std::unique_ptr<cricket::SctpTransportInternal> CreateSctpTransport(
      const Environment& env,
      rtc::PacketTransportInternal* channel) = 0;
};

}  // namespace webrtc

#endif  // API_TRANSPORT_SCTP_TRANSPORT_FACTORY_INTERFACE_H_
