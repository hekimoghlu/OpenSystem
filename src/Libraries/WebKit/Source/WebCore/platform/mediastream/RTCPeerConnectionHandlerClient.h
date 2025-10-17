/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#pragma once

#if ENABLE(WEB_RTC)

#include "RTCIceConnectionState.h"
#include "RTCIceGatheringState.h"
#include "RTCSignalingState.h"
#include <wtf/RefPtr.h>

namespace WebCore {

class MediaStreamPrivate;
class RTCDataChannelHandler;
class RTCIceCandidateDescriptor;

class RTCPeerConnectionHandlerClient {
public:
    virtual ~RTCPeerConnectionHandlerClient() = default;

    virtual void negotiationNeeded() = 0;
    virtual void didGenerateIceCandidate(RefPtr<RTCIceCandidateDescriptor>&&) = 0;
    virtual void didChangeSignalingState(RTCSignalingState) = 0;
    virtual void didChangeIceGatheringState(RTCIceGatheringState) = 0;
    virtual void didChangeIceConnectionState(RTCIceConnectionState) = 0;
    virtual void didAddRemoteStream(RefPtr<MediaStreamPrivate>&&) = 0;
    virtual void didRemoveRemoteStream(MediaStreamPrivate*) = 0;
    virtual void didAddRemoteDataChannel(std::unique_ptr<RTCDataChannelHandler>) = 0;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
