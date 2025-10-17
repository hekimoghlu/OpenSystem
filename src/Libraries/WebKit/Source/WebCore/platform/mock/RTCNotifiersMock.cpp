/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#include "config.h"

#if ENABLE(WEB_RTC)

#include "RTCNotifiersMock.h"

#include "RTCDataChannelHandlerMock.h"
#include "RTCSessionDescriptionDescriptor.h"
#include "RTCSessionDescriptionRequest.h"
#include "RTCVoidRequest.h"

namespace WebCore {

SessionRequestNotifier::SessionRequestNotifier(RefPtr<RTCSessionDescriptionRequest>&& request, RefPtr<RTCSessionDescriptionDescriptor>&& descriptor, const String& errorName)
    : m_request(WTFMove(request))
    , m_descriptor(WTFMove(descriptor))
    , m_errorName(errorName)
{
}

void SessionRequestNotifier::fire()
{
    if (m_descriptor)
        m_request->requestSucceeded(*m_descriptor);
    else
        m_request->requestFailed(m_errorName);
}

VoidRequestNotifier::VoidRequestNotifier(RefPtr<RTCVoidRequest>&& request, bool success, const String& errorName)
    : m_request(WTFMove(request))
    , m_success(success)
    , m_errorName(errorName)
{
}

void VoidRequestNotifier::fire()
{
    if (m_success)
        m_request->requestSucceeded();
    else
        m_request->requestFailed(m_errorName);
}

IceConnectionNotifier::IceConnectionNotifier(RTCPeerConnectionHandlerClient* client, RTCIceConnectionState connectionState, RTCIceGatheringState gatheringState)
    : m_client(client)
    , m_connectionState(connectionState)
    , m_gatheringState(gatheringState)
{
}

void IceConnectionNotifier::fire()
{
    m_client->didChangeIceGatheringState(m_gatheringState);
    m_client->didChangeIceConnectionState(m_connectionState);
}

SignalingStateNotifier::SignalingStateNotifier(RTCPeerConnectionHandlerClient* client, RTCSignalingState signalingState)
    : m_client(client)
    , m_signalingState(signalingState)
{
}

void SignalingStateNotifier::fire()
{
    m_client->didChangeSignalingState(m_signalingState);
}

RemoteDataChannelNotifier::RemoteDataChannelNotifier(RTCPeerConnectionHandlerClient* client)
    : m_client(client)
{
}

void RemoteDataChannelNotifier::fire()
{
    m_client->didAddRemoteDataChannel(makeUnique<RTCDataChannelHandlerMock>("RTCDataChannelHandlerMock"_s, RTCDataChannelInit()));
}

DataChannelStateNotifier::DataChannelStateNotifier(RTCDataChannelHandlerClient* client, RTCDataChannelState state)
    : m_client(client)
    , m_state(state)
{
}

void DataChannelStateNotifier::fire()
{
    m_client->didChangeReadyState(m_state);
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
