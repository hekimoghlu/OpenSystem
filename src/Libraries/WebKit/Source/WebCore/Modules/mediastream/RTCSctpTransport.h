/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

#include "ActiveDOMObject.h"
#include "Event.h"
#include "EventTarget.h"
#include "RTCSctpTransportBackend.h"

namespace WebCore {

class RTCDtlsTransport;

class RTCSctpTransport final : public RefCounted<RTCSctpTransport>, public ActiveDOMObject, public EventTarget, public RTCSctpTransportBackendClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCSctpTransport);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<RTCSctpTransport> create(ScriptExecutionContext&, UniqueRef<RTCSctpTransportBackend>&&, Ref<RTCDtlsTransport>&&);
    ~RTCSctpTransport();

    RTCDtlsTransport& transport() { return m_transport.get(); }
    RTCSctpTransportState state() const { return m_state; }
    double maxMessageSize() const { return m_maxMessageSize.value_or(std::numeric_limits<double>::infinity()); }
    std::optional<unsigned short>  maxChannels() const { return m_maxChannels; }

    void updateMaxMessageSize(std::optional<double>);

    const RTCSctpTransportBackend& backend() const { return m_backend.get(); }

private:
    RTCSctpTransport(ScriptExecutionContext&, UniqueRef<RTCSctpTransportBackend>&&, Ref<RTCDtlsTransport>&&);

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::RTCSctpTransport; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject
    void stop() final;
    bool virtualHasPendingActivity() const final;

    // RTCSctpTransport::Client
    void onStateChanged(RTCSctpTransportState, std::optional<double>, std::optional<unsigned short>) final;

    UniqueRef<RTCSctpTransportBackend> m_backend;
    Ref<RTCDtlsTransport> m_transport;
    RTCSctpTransportState m_state { RTCSctpTransportState::Connecting };
    std::optional<double> m_maxMessageSize;
    std::optional<unsigned short> m_maxChannels;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
