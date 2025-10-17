/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#ifndef API_ICE_TRANSPORT_FACTORY_H_
#define API_ICE_TRANSPORT_FACTORY_H_

#include "api/ice_transport_interface.h"
#include "api/scoped_refptr.h"
#include "rtc_base/system/rtc_export.h"

namespace cricket {
class PortAllocator;  // IWYU pragma: keep
}  // namespace cricket

namespace webrtc {

// Static factory for an IceTransport object that can be created
// without using a webrtc::PeerConnection.
// The returned object must be accessed and destroyed on the thread that
// created it.
// The PortAllocator must outlive the created IceTransportInterface object.
// TODO(steveanton): Remove in favor of the overload that takes
// IceTransportInit.
RTC_EXPORT rtc::scoped_refptr<IceTransportInterface> CreateIceTransport(
    cricket::PortAllocator* port_allocator);

// Static factory for an IceTransport object that can be created
// without using a webrtc::PeerConnection.
// The returned object must be accessed and destroyed on the thread that
// created it.
// `init.port_allocator()` is required and must outlive the created
//     IceTransportInterface object.
// `init.async_resolver_factory()` and `init.event_log()` are optional, but if
//     provided must outlive the created IceTransportInterface object.
RTC_EXPORT rtc::scoped_refptr<IceTransportInterface> CreateIceTransport(
    IceTransportInit);

}  // namespace webrtc

#endif  // API_ICE_TRANSPORT_FACTORY_H_
