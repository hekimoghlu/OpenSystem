/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#include "LibWebRTCDtlsTransportBackend.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCIceTransportBackend.h"
#include "LibWebRTCProvider.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <webrtc/api/dtls_transport_interface.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static inline RTCDtlsTransportState toRTCDtlsTransportState(webrtc::DtlsTransportState state)
{
    switch (state) {
    case webrtc::DtlsTransportState::kNew:
        return RTCDtlsTransportState::New;
    case webrtc::DtlsTransportState::kConnecting:
        return RTCDtlsTransportState::Connecting;
    case webrtc::DtlsTransportState::kConnected:
        return RTCDtlsTransportState::Connected;
    case webrtc::DtlsTransportState::kClosed:
        return RTCDtlsTransportState::Closed;
    case webrtc::DtlsTransportState::kFailed:
        return RTCDtlsTransportState::Failed;
    case webrtc::DtlsTransportState::kNumValues:
        ASSERT_NOT_REACHED();
        return RTCDtlsTransportState::Failed;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

class LibWebRTCDtlsTransportBackendObserver final : public ThreadSafeRefCounted<LibWebRTCDtlsTransportBackendObserver>, public webrtc::DtlsTransportObserverInterface {
public:
    static Ref<LibWebRTCDtlsTransportBackendObserver> create(RTCDtlsTransportBackendClient& client, rtc::scoped_refptr<webrtc::DtlsTransportInterface>& backend) { return adoptRef(*new LibWebRTCDtlsTransportBackendObserver(client, backend)); }

    void start();
    void stop();

private:
    LibWebRTCDtlsTransportBackendObserver(RTCDtlsTransportBackendClient&, rtc::scoped_refptr<webrtc::DtlsTransportInterface>&);

    void OnStateChange(webrtc::DtlsTransportInformation) final;
    void OnError(webrtc::RTCError) final;

    void updateState(webrtc::DtlsTransportInformation&&);

    rtc::scoped_refptr<webrtc::DtlsTransportInterface> m_backend;
    WeakPtr<RTCDtlsTransportBackendClient> m_client;
};

LibWebRTCDtlsTransportBackendObserver::LibWebRTCDtlsTransportBackendObserver(RTCDtlsTransportBackendClient& client, rtc::scoped_refptr<webrtc::DtlsTransportInterface>& backend)
    : m_backend(backend)
    , m_client(client)
{
    ASSERT(m_backend);
}

void LibWebRTCDtlsTransportBackendObserver::updateState(webrtc::DtlsTransportInformation&& info)
{
    if (!m_client)
        return;

    Vector<rtc::Buffer> certificates;
    if (auto* remoteCertificates = info.remote_ssl_certificates()) {
        for (size_t i = 0; i < remoteCertificates->GetSize(); ++i) {
            rtc::Buffer certificate;
            remoteCertificates->Get(i).ToDER(&certificate);
            certificates.append(WTFMove(certificate));
        }
    }
    m_client->onStateChanged(toRTCDtlsTransportState(info.state()), map(certificates, [](auto& certificate) -> Ref<JSC::ArrayBuffer> {
        return JSC::ArrayBuffer::create(certificate);
    }));
}

void LibWebRTCDtlsTransportBackendObserver::start()
{
    LibWebRTCProvider::callOnWebRTCNetworkThread([this, protectedThis = Ref { *this }]() mutable {
        m_backend->RegisterObserver(this);
        callOnMainThread([protectedThis = WTFMove(protectedThis), info = m_backend->Information()]() mutable {
            protectedThis->updateState(WTFMove(info));
        });
    });
}

void LibWebRTCDtlsTransportBackendObserver::stop()
{
    m_client = nullptr;
    LibWebRTCProvider::callOnWebRTCNetworkThread([protectedThis = Ref { *this }] {
        protectedThis->m_backend->UnregisterObserver();
    });
}

void LibWebRTCDtlsTransportBackendObserver::OnStateChange(webrtc::DtlsTransportInformation info)
{
    callOnMainThread([protectedThis = Ref { *this }, info = WTFMove(info)]() mutable {
        protectedThis->updateState(WTFMove(info));
    });
}

void LibWebRTCDtlsTransportBackendObserver::OnError(webrtc::RTCError)
{
    callOnMainThread([protectedThis = Ref { *this }] {
        if (protectedThis->m_client)
            protectedThis->m_client->onError();
    });
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCDtlsTransportBackend);

LibWebRTCDtlsTransportBackend::LibWebRTCDtlsTransportBackend(rtc::scoped_refptr<webrtc::DtlsTransportInterface>&& backend)
    : m_backend(WTFMove(backend))
{
    ASSERT(m_backend);
}

LibWebRTCDtlsTransportBackend::~LibWebRTCDtlsTransportBackend()
{
    if (m_observer)
        m_observer->stop();
}

UniqueRef<RTCIceTransportBackend> LibWebRTCDtlsTransportBackend::iceTransportBackend()
{
    return makeUniqueRef<LibWebRTCIceTransportBackend>(m_backend->ice_transport());
}

void LibWebRTCDtlsTransportBackend::registerClient(RTCDtlsTransportBackendClient& client)
{
    ASSERT(!m_observer);
    m_observer = LibWebRTCDtlsTransportBackendObserver::create(client, m_backend);
    m_observer->start();
}

void LibWebRTCDtlsTransportBackend::unregisterClient()
{
    ASSERT(m_observer);
    m_observer->stop();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
