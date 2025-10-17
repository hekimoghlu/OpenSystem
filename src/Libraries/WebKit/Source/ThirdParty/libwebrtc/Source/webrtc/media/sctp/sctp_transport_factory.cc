/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#include "media/sctp/sctp_transport_factory.h"

#include "api/environment/environment.h"
#include "rtc_base/system/unused.h"

#ifdef WEBRTC_HAVE_DCSCTP
#include "media/sctp/dcsctp_transport.h"  // nogncheck
#endif

namespace cricket {

SctpTransportFactory::SctpTransportFactory(rtc::Thread* network_thread)
    : network_thread_(network_thread) {
  RTC_UNUSED(network_thread_);
}

std::unique_ptr<SctpTransportInternal>
SctpTransportFactory::CreateSctpTransport(
    const webrtc::Environment& env,
    rtc::PacketTransportInternal* transport) {
  std::unique_ptr<SctpTransportInternal> result;
#ifdef WEBRTC_HAVE_DCSCTP
  result = std::unique_ptr<SctpTransportInternal>(
      new webrtc::DcSctpTransport(env, network_thread_, transport));
#endif
  return result;
}

}  // namespace cricket
