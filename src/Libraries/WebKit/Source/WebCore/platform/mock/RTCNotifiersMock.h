/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

#include "RTCDataChannelHandlerClient.h"
#include "RTCPeerConnectionHandlerClient.h"
#include "TimerEventBasedMock.h"
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCSessionDescriptionDescriptor;
class RTCSessionDescriptionRequest;
class RTCVoidRequest;

class SessionRequestNotifier : public MockNotifier {
public:
    SessionRequestNotifier(RefPtr<RTCSessionDescriptionRequest>&&, RefPtr<RTCSessionDescriptionDescriptor>&&, const String& = emptyString());

    void fire() override;

private:
    RefPtr<RTCSessionDescriptionRequest> m_request;
    RefPtr<RTCSessionDescriptionDescriptor> m_descriptor;
    String m_errorName;
};

class VoidRequestNotifier : public MockNotifier {
public:
    VoidRequestNotifier(RefPtr<RTCVoidRequest>&&, bool, const String& = emptyString());

    void fire() override;

private:
    RefPtr<RTCVoidRequest> m_request;
    bool m_success;
    String m_errorName;
};

class IceConnectionNotifier : public MockNotifier {
public:
    IceConnectionNotifier(RTCPeerConnectionHandlerClient*, RTCIceConnectionState, RTCIceGatheringState);

    void fire() override;

private:
    RTCPeerConnectionHandlerClient* m_client;
    RTCIceConnectionState m_connectionState;
    RTCIceGatheringState m_gatheringState;
};

class SignalingStateNotifier : public MockNotifier {
public:
    SignalingStateNotifier(RTCPeerConnectionHandlerClient*, RTCSignalingState);

    void fire() override;

private:
    RTCPeerConnectionHandlerClient* m_client;
    RTCSignalingState m_signalingState;
};

class RemoteDataChannelNotifier : public MockNotifier {
public:
    RemoteDataChannelNotifier(RTCPeerConnectionHandlerClient*);

    void fire() override;

private:
    RTCPeerConnectionHandlerClient* m_client;
};

class DataChannelStateNotifier : public MockNotifier {
public:
    DataChannelStateNotifier(RTCDataChannelHandlerClient*, RTCDataChannelState);

    void fire() override;

private:
    RTCDataChannelHandlerClient* m_client;
    RTCDataChannelState m_state;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
