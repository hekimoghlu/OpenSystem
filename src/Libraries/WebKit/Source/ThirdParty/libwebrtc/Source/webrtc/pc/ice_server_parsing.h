/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#ifndef PC_ICE_SERVER_PARSING_H_
#define PC_ICE_SERVER_PARSING_H_

#include <vector>

#include "api/peer_connection_interface.h"
#include "api/rtc_error.h"
#include "p2p/base/port.h"
#include "p2p/base/port_allocator.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Parses the URLs for each server in `servers` to build `stun_servers` and
// `turn_servers`. Can return SYNTAX_ERROR if the URL is malformed, or
// INVALID_PARAMETER if a TURN server is missing `username` or `password`.
//
// Intended to be used to convert/validate the servers passed into a
// PeerConnection through RTCConfiguration.
RTC_EXPORT RTCError
ParseIceServersOrError(const PeerConnectionInterface::IceServers& servers,
                       cricket::ServerAddresses* stun_servers,
                       std::vector<cricket::RelayServerConfig>* turn_servers);

}  // namespace webrtc

#endif  // PC_ICE_SERVER_PARSING_H_
