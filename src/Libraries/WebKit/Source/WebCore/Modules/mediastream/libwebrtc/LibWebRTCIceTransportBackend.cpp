/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include "LibWebRTCIceTransportBackend.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCProvider.h"
#include "LibWebRTCUtils.h"
#include "RTCIceCandidate.h"
#include <wtf/TZoneMallocInlines.h>

ALLOW_UNUSED_PARAMETERS_BEGIN

#include <webrtc/api/ice_transport_interface.h>
#include <webrtc/p2p/base/ice_transport_internal.h>

ALLOW_UNUSED_PARAMETERS_END

#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

static inline RTCIceTransportState toRTCIceTransportState(webrtc::IceTransportState state)
{
    switch (state) {
    case webrtc::IceTransportState::kNew:
        return RTCIceTransportState::New;
    case webrtc::IceTransportState::kChecking:
        return RTCIceTransportState::Checking;
    case webrtc::IceTransportState::kConnected:
        return RTCIceTransportState::Connected;
    case webrtc::IceTransportState::kFailed:
        return RTCIceTransportState::Failed;
    case webrtc::IceTransportState::kCompleted:
        return RTCIceTransportState::Completed;
    case webrtc::IceTransportState::kDisconnected:
        return RTCIceTransportState::Disconnected;
    case webrtc::IceTransportState::kClosed:
        return RTCIceTransportState::Closed;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

static inline RTCIceGatheringState toRTCIceGatheringState(cricket::IceGatheringState state)
{
    switch (state) {
    case cricket::IceGatheringState::kIceGatheringNew:
        return RTCIceGatheringState::New;
    case cricket::IceGatheringState::kIceGatheringGathering:
        return RTCIceGatheringState::Gathering;
    case cricket::IceGatheringState::kIceGatheringComplete:
        return RTCIceGatheringState::Complete;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

class LibWebRTCIceTransportBackendObserver final : public ThreadSafeRefCounted<LibWebRTCIceTransportBackendObserver>, public sigslot::has_slots<> {
public:
    static Ref<LibWebRTCIceTransportBackendObserver> create(RTCIceTransportBackendClient& client, rtc::scoped_refptr<webrtc::IceTransportInterface> backend) { return adoptRef(*new LibWebRTCIceTransportBackendObserver(client, WTFMove(backend))); }

    void start();
    void stop();

private:
    LibWebRTCIceTransportBackendObserver(RTCIceTransportBackendClient&, rtc::scoped_refptr<webrtc::IceTransportInterface>&&);

    void onIceTransportStateChanged(cricket::IceTransportInternal*);
    void onGatheringStateChanged(cricket::IceTransportInternal*);
    void onNetworkRouteChanged(std::optional<rtc::NetworkRoute>);

    void processSelectedCandidatePairChanged(const cricket::Candidate&, const cricket::Candidate&);

    rtc::scoped_refptr<webrtc::IceTransportInterface> m_backend;
    WeakPtr<RTCIceTransportBackendClient> m_client;
};

LibWebRTCIceTransportBackendObserver::LibWebRTCIceTransportBackendObserver(RTCIceTransportBackendClient& client, rtc::scoped_refptr<webrtc::IceTransportInterface>&& backend)
    : m_backend(WTFMove(backend))
    , m_client(client)
{
    ASSERT(m_backend);
}

void LibWebRTCIceTransportBackendObserver::start()
{
    LibWebRTCProvider::callOnWebRTCNetworkThread([this, protectedThis = Ref { *this }]() mutable {
        auto* internal = m_backend->internal();
        if (!internal)
            return;
        internal->SignalIceTransportStateChanged.connect(this, &LibWebRTCIceTransportBackendObserver::onIceTransportStateChanged);
        internal->AddGatheringStateCallback(this, [this](auto* transport) { onGatheringStateChanged(transport); });
        internal->SignalNetworkRouteChanged.connect(this, &LibWebRTCIceTransportBackendObserver::onNetworkRouteChanged);

        auto transportState = internal->GetIceTransportState();
        // We start observing a bit late and might miss the checking state. Synthesize it as needed.
        if (transportState > webrtc::IceTransportState::kChecking && transportState != webrtc::IceTransportState::kClosed) {
            callOnMainThread([protectedThis = Ref { *this }] {
                if (protectedThis->m_client)
                    protectedThis->m_client->onStateChanged(RTCIceTransportState::Checking);
            });
        }
        callOnMainThread([protectedThis = Ref { *this }, transportState, gatheringState = internal->gathering_state()] {
            if (!protectedThis->m_client)
                return;
            protectedThis->m_client->onStateChanged(toRTCIceTransportState(transportState));
            protectedThis->m_client->onGatheringStateChanged(toRTCIceGatheringState(gatheringState));
        });

        if (auto candidatePair = internal->GetSelectedCandidatePair())
            processSelectedCandidatePairChanged(candidatePair->local, candidatePair->remote);
    });
}

void LibWebRTCIceTransportBackendObserver::stop()
{
    m_client = nullptr;
    LibWebRTCProvider::callOnWebRTCNetworkThread([this, protectedThis = Ref { *this }] {
        auto* internal = m_backend->internal();
        if (!internal)
            return;
        internal->SignalIceTransportStateChanged.disconnect(this);
        internal->RemoveGatheringStateCallback(this);
        internal->SignalNetworkRouteChanged.disconnect(this);
    });
}

void LibWebRTCIceTransportBackendObserver::onIceTransportStateChanged(cricket::IceTransportInternal* internal)
{
    callOnMainThread([protectedThis = Ref { *this }, state = internal->GetIceTransportState()] {
        if (protectedThis->m_client)
            protectedThis->m_client->onStateChanged(toRTCIceTransportState(state));
    });
}

void LibWebRTCIceTransportBackendObserver::onGatheringStateChanged(cricket::IceTransportInternal* internal)
{
    callOnMainThread([protectedThis = Ref { *this }, state = internal->gathering_state()] {
        if (protectedThis->m_client)
            protectedThis->m_client->onGatheringStateChanged(toRTCIceGatheringState(state));
    });
}

void LibWebRTCIceTransportBackendObserver::onNetworkRouteChanged(std::optional<rtc::NetworkRoute>)
{
    if (auto selectedPair = m_backend->internal()->GetSelectedCandidatePair())
        processSelectedCandidatePairChanged(selectedPair->local_candidate(), selectedPair->remote_candidate());
}

void LibWebRTCIceTransportBackendObserver::processSelectedCandidatePairChanged(const cricket::Candidate& local, const cricket::Candidate& remote)
{
    callOnMainThread([protectedThis = Ref { *this }, localSdp = fromStdString(local.ToString()).isolatedCopy(), remoteSdp = fromStdString(remote.ToString()).isolatedCopy(), localFields = convertIceCandidate(local).isolatedCopy(), remoteFields = convertIceCandidate(remote).isolatedCopy()]() mutable {
        if (!protectedThis->m_client)
            return;

        auto local = RTCIceCandidate::create(localSdp, emptyString(), WTFMove(localFields));
        auto remote = RTCIceCandidate::create(remoteSdp, emptyString(), WTFMove(remoteFields));
        protectedThis->m_client->onSelectedCandidatePairChanged(WTFMove(local), WTFMove(remote));
    });
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCIceTransportBackend);

LibWebRTCIceTransportBackend::LibWebRTCIceTransportBackend(rtc::scoped_refptr<webrtc::IceTransportInterface>&& backend)
    : m_backend(WTFMove(backend))
{
    ASSERT(m_backend);
}

LibWebRTCIceTransportBackend::~LibWebRTCIceTransportBackend()
{
}

void LibWebRTCIceTransportBackend::registerClient(RTCIceTransportBackendClient& client)
{
    ASSERT(!m_observer);
    m_observer = LibWebRTCIceTransportBackendObserver::create(client, m_backend);
    m_observer->start();
}

void LibWebRTCIceTransportBackend::unregisterClient()
{
    ASSERT(m_observer);
    m_observer->stop();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
