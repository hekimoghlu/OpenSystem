/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#include "LibWebRTCSctpTransportBackend.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCDtlsTransportBackend.h"
#include "LibWebRTCProvider.h"
#include <wtf/TZoneMallocInlines.h>

ALLOW_UNUSED_PARAMETERS_BEGIN

#include <webrtc/api/sctp_transport_interface.h>

ALLOW_UNUSED_PARAMETERS_END

namespace WebCore {

static inline RTCSctpTransportState toRTCSctpTransportState(webrtc::SctpTransportState state)
{
    switch (state) {
    case webrtc::SctpTransportState::kNew:
    case webrtc::SctpTransportState::kConnecting:
        return RTCSctpTransportState::Connecting;
    case webrtc::SctpTransportState::kConnected:
        return RTCSctpTransportState::Connected;
    case webrtc::SctpTransportState::kClosed:
        return RTCSctpTransportState::Closed;
    case webrtc::SctpTransportState::kNumValues:
        ASSERT_NOT_REACHED();
        return RTCSctpTransportState::Connecting;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

class LibWebRTCSctpTransportBackendObserver final : public ThreadSafeRefCounted<LibWebRTCSctpTransportBackendObserver>, public webrtc::SctpTransportObserverInterface {
public:
    static Ref<LibWebRTCSctpTransportBackendObserver> create(RTCSctpTransportBackendClient& client, rtc::scoped_refptr<webrtc::SctpTransportInterface>& backend) { return adoptRef(*new LibWebRTCSctpTransportBackendObserver(client, backend)); }

    void start();
    void stop();

private:
    LibWebRTCSctpTransportBackendObserver(RTCSctpTransportBackendClient&, rtc::scoped_refptr<webrtc::SctpTransportInterface>&);

    void OnStateChange(webrtc::SctpTransportInformation) final;

    void updateState(webrtc::SctpTransportInformation&&);

    rtc::scoped_refptr<webrtc::SctpTransportInterface> m_backend;
    WeakPtr<RTCSctpTransportBackendClient> m_client;
};

LibWebRTCSctpTransportBackendObserver::LibWebRTCSctpTransportBackendObserver(RTCSctpTransportBackendClient& client, rtc::scoped_refptr<webrtc::SctpTransportInterface>& backend)
    : m_backend(backend)
    , m_client(client)
{
    ASSERT(m_backend);
}

void LibWebRTCSctpTransportBackendObserver::updateState(webrtc::SctpTransportInformation&& info)
{
    if (!m_client)
        return;

    std::optional<unsigned short> maxChannels;
    if (auto value = info.MaxChannels())
        maxChannels = *value;
    std::optional<double> maxMessageSize;
    if (info.MaxMessageSize())
        maxMessageSize = *info.MaxMessageSize();
    m_client->onStateChanged(toRTCSctpTransportState(info.state()), maxMessageSize, maxChannels);
}

void LibWebRTCSctpTransportBackendObserver::start()
{
    LibWebRTCProvider::callOnWebRTCNetworkThread([this, protectedThis = Ref { *this }]() mutable {
        m_backend->RegisterObserver(this);
        callOnMainThread([protectedThis = WTFMove(protectedThis), info = m_backend->Information()]() mutable {
            protectedThis->updateState(WTFMove(info));
        });
    });
}

void LibWebRTCSctpTransportBackendObserver::stop()
{
    m_client = nullptr;
    LibWebRTCProvider::callOnWebRTCNetworkThread([protectedThis = Ref { *this }] {
        protectedThis->m_backend->UnregisterObserver();
    });
}

void LibWebRTCSctpTransportBackendObserver::OnStateChange(webrtc::SctpTransportInformation info)
{
    callOnMainThread([protectedThis = Ref { *this }, info = WTFMove(info)]() mutable {
        protectedThis->updateState(WTFMove(info));
    });
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCSctpTransportBackend);

LibWebRTCSctpTransportBackend::LibWebRTCSctpTransportBackend(rtc::scoped_refptr<webrtc::SctpTransportInterface>&& backend, rtc::scoped_refptr<webrtc::DtlsTransportInterface>&& dtlsBackend)
    : m_backend(WTFMove(backend))
    , m_dtlsBackend(WTFMove(dtlsBackend))
{
    ASSERT(m_backend);
}

LibWebRTCSctpTransportBackend::~LibWebRTCSctpTransportBackend()
{
    if (m_observer)
        m_observer->stop();
}

UniqueRef<RTCDtlsTransportBackend> LibWebRTCSctpTransportBackend::dtlsTransportBackend()
{
    return makeUniqueRef<LibWebRTCDtlsTransportBackend>(rtc::scoped_refptr<webrtc::DtlsTransportInterface> { m_dtlsBackend.get() });
}

void LibWebRTCSctpTransportBackend::registerClient(RTCSctpTransportBackendClient& client)
{
    ASSERT(!m_observer);
    m_observer = LibWebRTCSctpTransportBackendObserver::create(client, m_backend);
    m_observer->start();
}

void LibWebRTCSctpTransportBackend::unregisterClient()
{
    ASSERT(m_observer);
    m_observer->stop();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
