/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#include "RTCDtlsTransport.h"

#if ENABLE(WEB_RTC)

#include "Blob.h"
#include "ContextDestructionObserverInlines.h"
#include "EventNames.h"
#include "Logging.h"
#include "NotImplemented.h"
#include "RTCDtlsTransportBackend.h"
#include "RTCIceTransport.h"
#include "RTCPeerConnection.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCDtlsTransport);

Ref<RTCDtlsTransport> RTCDtlsTransport::create(ScriptExecutionContext& context, UniqueRef<RTCDtlsTransportBackend>&& backend, Ref<RTCIceTransport>&& iceTransport)
{
    auto result = adoptRef(*new RTCDtlsTransport(context, WTFMove(backend), WTFMove(iceTransport)));
    result->suspendIfNeeded();
    return result;
}

RTCDtlsTransport::RTCDtlsTransport(ScriptExecutionContext& context, UniqueRef<RTCDtlsTransportBackend>&& backend, Ref<RTCIceTransport>&& iceTransport)
    : ActiveDOMObject(&context)
    , m_backend(WTFMove(backend))
    , m_iceTransport(WTFMove(iceTransport))
{
    m_backend->registerClient(*this);
}

RTCDtlsTransport::~RTCDtlsTransport()
{
    m_backend->unregisterClient();
}

Vector<Ref<JSC::ArrayBuffer>> RTCDtlsTransport::getRemoteCertificates()
{
    return m_remoteCertificates;
}

void RTCDtlsTransport::stop()
{
    m_state = RTCDtlsTransportState::Closed;
}

bool RTCDtlsTransport::virtualHasPendingActivity() const
{
    return m_state != RTCDtlsTransportState::Closed && hasEventListeners();
}

void RTCDtlsTransport::onStateChanged(RTCDtlsTransportState state, Vector<Ref<JSC::ArrayBuffer>>&& certificates)
{
    queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this, state, certificates = WTFMove(certificates)]() mutable {
        if (m_state == RTCDtlsTransportState::Closed)
            return;

        if (m_remoteCertificates != certificates)
            m_remoteCertificates = WTFMove(certificates);

        if (m_state != state) {
            m_state = state;
            if (RefPtr connection = m_iceTransport->connection())
                connection->updateConnectionState();
            dispatchEvent(Event::create(eventNames().statechangeEvent, Event::CanBubble::Yes, Event::IsCancelable::No));
        }
    });
}

void RTCDtlsTransport::onError()
{
    onStateChanged(RTCDtlsTransportState::Failed, { });
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
