/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
#include "RTCSctpTransport.h"

#if ENABLE(WEB_RTC)

#include "ContextDestructionObserverInlines.h"
#include "EventNames.h"
#include "Logging.h"
#include "RTCDtlsTransport.h"
#include "RTCSctpTransportBackend.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCSctpTransport);

Ref<RTCSctpTransport> RTCSctpTransport::create(ScriptExecutionContext& context, UniqueRef<RTCSctpTransportBackend>&& backend, Ref<RTCDtlsTransport>&& transport)
{
    auto result = adoptRef(*new RTCSctpTransport(context, WTFMove(backend), WTFMove(transport)));
    result->suspendIfNeeded();
    return result;
}

RTCSctpTransport::RTCSctpTransport(ScriptExecutionContext& context, UniqueRef<RTCSctpTransportBackend>&& backend, Ref<RTCDtlsTransport >&& transport)
    : ActiveDOMObject(&context)
    , m_backend(WTFMove(backend))
    , m_transport(WTFMove(transport))
{
    m_backend->registerClient(*this);
}

RTCSctpTransport::~RTCSctpTransport()
{
    m_backend->unregisterClient();
}

void RTCSctpTransport::stop()
{
    m_state = RTCSctpTransportState::Closed;
}

bool RTCSctpTransport::virtualHasPendingActivity() const
{
    return m_state != RTCSctpTransportState::Closed && hasEventListeners();
}

void RTCSctpTransport::onStateChanged(RTCSctpTransportState state, std::optional<double> maxMessageSize, std::optional<unsigned short> maxChannels)
{
    queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this, state, maxMessageSize, maxChannels]() mutable {
        if (m_state == RTCSctpTransportState::Closed)
            return;

        m_maxMessageSize = maxMessageSize;
        if (maxChannels)
            m_maxChannels = *maxChannels;

        if (m_state != state) {
            m_state = state;
            dispatchEvent(Event::create(eventNames().statechangeEvent, Event::CanBubble::Yes, Event::IsCancelable::No));
        }
    });
}

void RTCSctpTransport::updateMaxMessageSize(std::optional<double> maxMessageSize)
{
    m_maxMessageSize = maxMessageSize;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
