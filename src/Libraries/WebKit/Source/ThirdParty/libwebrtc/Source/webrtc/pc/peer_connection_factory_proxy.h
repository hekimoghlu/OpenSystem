/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
#ifndef PC_PEER_CONNECTION_FACTORY_PROXY_H_
#define PC_PEER_CONNECTION_FACTORY_PROXY_H_

#include <memory>
#include <string>
#include <utility>

#include "api/peer_connection_interface.h"
#include "pc/proxy.h"

namespace webrtc {

// TODO(deadbeef): Move this to .cc file. What threads methods are called on is
// an implementation detail.
BEGIN_PROXY_MAP(PeerConnectionFactory)
PROXY_PRIMARY_THREAD_DESTRUCTOR()
PROXY_METHOD1(void, SetOptions, const Options&)
PROXY_METHOD2(RTCErrorOr<rtc::scoped_refptr<PeerConnectionInterface>>,
              CreatePeerConnectionOrError,
              const PeerConnectionInterface::RTCConfiguration&,
              PeerConnectionDependencies)
PROXY_CONSTMETHOD1(RtpCapabilities,
                   GetRtpSenderCapabilities,
                   cricket::MediaType)
PROXY_CONSTMETHOD1(RtpCapabilities,
                   GetRtpReceiverCapabilities,
                   cricket::MediaType)
PROXY_METHOD1(rtc::scoped_refptr<MediaStreamInterface>,
              CreateLocalMediaStream,
              const std::string&)
PROXY_METHOD1(rtc::scoped_refptr<AudioSourceInterface>,
              CreateAudioSource,
              const cricket::AudioOptions&)
PROXY_METHOD2(rtc::scoped_refptr<VideoTrackInterface>,
              CreateVideoTrack,
              rtc::scoped_refptr<VideoTrackSourceInterface>,
              absl::string_view)
PROXY_METHOD2(rtc::scoped_refptr<AudioTrackInterface>,
              CreateAudioTrack,
              const std::string&,
              AudioSourceInterface*)
PROXY_SECONDARY_METHOD2(bool, StartAecDump, FILE*, int64_t)
PROXY_SECONDARY_METHOD0(void, StopAecDump)
END_PROXY_MAP(PeerConnectionFactory)

}  // namespace webrtc

#endif  // PC_PEER_CONNECTION_FACTORY_PROXY_H_
