/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#include "EventTarget.h"
#include "RTCIceCandidate.h"
#include "RTCIceGatheringState.h"
#include "RTCIceTransportBackend.h"
#include "RTCIceTransportState.h"
#include "RTCPeerConnection.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class RTCPeerConnection;

class RTCIceTransport : public RefCounted<RTCIceTransport>, public ActiveDOMObject, public EventTarget, public RTCIceTransportBackendClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCIceTransport);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<RTCIceTransport> create(ScriptExecutionContext&, UniqueRef<RTCIceTransportBackend>&&, RTCPeerConnection&);
    ~RTCIceTransport();

    RTCIceTransportState state() const { return m_transportState; }
    RTCIceGatheringState gatheringState() const { return m_gatheringState; }

    const RTCIceTransportBackend& backend() const { return m_backend.get(); }
    RefPtr<RTCPeerConnection> connection() const { return m_connection.get(); }

    struct CandidatePair {
        RefPtr<RTCIceCandidate> local;
        RefPtr<RTCIceCandidate> remote;
    };
    std::optional<CandidatePair> getSelectedCandidatePair();

private:
    RTCIceTransport(ScriptExecutionContext&, UniqueRef<RTCIceTransportBackend>&&, RTCPeerConnection&);

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::RTCIceTransport; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;

    // RTCIceTransportBackendClient
    void onStateChanged(RTCIceTransportState) final;
    void onGatheringStateChanged(RTCIceGatheringState) final;
    void onSelectedCandidatePairChanged(RefPtr<RTCIceCandidate>&&, RefPtr<RTCIceCandidate>&&) final;

    bool m_isStopped { false };
    UniqueRef<RTCIceTransportBackend> m_backend;
    WeakPtr<RTCPeerConnection, WeakPtrImplWithEventTargetData> m_connection;
    RTCIceTransportState m_transportState { RTCIceTransportState::New };
    RTCIceGatheringState m_gatheringState { RTCIceGatheringState::New };
    std::optional<CandidatePair> m_selectedCandidatePair;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
