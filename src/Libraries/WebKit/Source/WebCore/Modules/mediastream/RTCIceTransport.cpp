/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#include "RTCIceTransport.h"

#if ENABLE(WEB_RTC)

#include "ContextDestructionObserverInlines.h"
#include "Event.h"
#include "EventNames.h"
#include "RTCPeerConnection.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCIceTransport);

Ref<RTCIceTransport> RTCIceTransport::create(ScriptExecutionContext& context, UniqueRef<RTCIceTransportBackend>&& backend, RTCPeerConnection& connection)
{
    auto result = adoptRef(*new RTCIceTransport(context, WTFMove(backend), connection));
    result->suspendIfNeeded();
    return result;
}

RTCIceTransport::RTCIceTransport(ScriptExecutionContext& context, UniqueRef<RTCIceTransportBackend>&& backend, RTCPeerConnection& connection)
    : ActiveDOMObject(&context)
    , m_backend(WTFMove(backend))
    , m_connection(connection)
{
    m_backend->registerClient(*this);
}

RTCIceTransport::~RTCIceTransport()
{
    m_backend->unregisterClient();
}

void RTCIceTransport::stop()
{
    m_isStopped = true;
    m_transportState = RTCIceTransportState::Closed;
}

bool RTCIceTransport::virtualHasPendingActivity() const
{
    return m_transportState != RTCIceTransportState::Closed && hasEventListeners();
}

void RTCIceTransport::onStateChanged(RTCIceTransportState state)
{
    queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this, state]() mutable {
        if (m_isStopped)
            return;

        if (m_transportState == state)
            return;

        m_transportState = state;
        if (m_transportState == RTCIceTransportState::Failed)
            m_selectedCandidatePair = { };

        if (auto connection = this->connection())
            connection->processIceTransportStateChange(*this);
    });
}

void RTCIceTransport::onGatheringStateChanged(RTCIceGatheringState state)
{
    queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this, state]() mutable {
        if (m_isStopped)
            return;

        if (m_gatheringState == state)
            return;

        m_gatheringState = state;
        dispatchEvent(Event::create(eventNames().gatheringstatechangeEvent, Event::CanBubble::Yes, Event::IsCancelable::No));
    });
}

void RTCIceTransport::onSelectedCandidatePairChanged(RefPtr<RTCIceCandidate>&& local, RefPtr<RTCIceCandidate>&& remote)
{
    queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this, local = WTFMove(local), remote = WTFMove(remote)]() mutable {
        if (m_isStopped)
            return;

        m_selectedCandidatePair = CandidatePair { WTFMove(local), WTFMove(remote) };
        dispatchEvent(Event::create(eventNames().selectedcandidatepairchangeEvent, Event::CanBubble::Yes, Event::IsCancelable::No));
    });
}

std::optional<RTCIceTransport::CandidatePair> RTCIceTransport::getSelectedCandidatePair()
{
    if (m_transportState == RTCIceTransportState::Closed)
        return { };

    ASSERT(m_transportState != RTCIceTransportState::New || !m_selectedCandidatePair);
    return m_selectedCandidatePair;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
