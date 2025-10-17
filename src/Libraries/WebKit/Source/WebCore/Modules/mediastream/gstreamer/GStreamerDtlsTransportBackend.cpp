/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
#include "GStreamerDtlsTransportBackend.h"

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "GStreamerIceTransportBackend.h"
#include "GStreamerWebRTCUtils.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

GST_DEBUG_CATEGORY(webkit_webrtc_dtls_transport_debug);
#define GST_CAT_DEFAULT webkit_webrtc_dtls_transport_debug

class GStreamerDtlsTransportBackendObserver final : public ThreadSafeRefCounted<GStreamerDtlsTransportBackendObserver> {
public:
    static Ref<GStreamerDtlsTransportBackendObserver> create(RTCDtlsTransportBackendClient& client, GRefPtr<GstWebRTCDTLSTransport>&& backend) { return adoptRef(*new GStreamerDtlsTransportBackendObserver(client, WTFMove(backend))); }

    void start();
    void stop();

private:
    GStreamerDtlsTransportBackendObserver(RTCDtlsTransportBackendClient&, GRefPtr<GstWebRTCDTLSTransport>&&);

    void stateChanged();

    GRefPtr<GstWebRTCDTLSTransport> m_backend;
    WeakPtr<RTCDtlsTransportBackendClient> m_client;
};

GStreamerDtlsTransportBackendObserver::GStreamerDtlsTransportBackendObserver(RTCDtlsTransportBackendClient& client, GRefPtr<GstWebRTCDTLSTransport>&& backend)
    : m_backend(WTFMove(backend))
    , m_client(client)
{
    ASSERT(m_backend);
}

void GStreamerDtlsTransportBackendObserver::stateChanged()
{
    if (!m_client)
        return;

    callOnMainThread([this, protectedThis = Ref { *this }]() mutable {
        if (!m_client || !m_backend)
            return;

        GstWebRTCDTLSTransportState state;
        g_object_get(m_backend.get(), "state", &state, nullptr);

#ifndef GST_DISABLE_GST_DEBUG
        GUniquePtr<char> desc(g_enum_to_string(GST_TYPE_WEBRTC_DTLS_TRANSPORT_STATE, state));
        GST_DEBUG_OBJECT(m_backend.get(), "DTLS transport state changed to %s", desc.get());
#endif

        Vector<Ref<JSC::ArrayBuffer>> certificates;

        // Access to DTLS certificates is not memory-safe in GStreamer versions older than 1.22.3.
        // See also: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/commit/d9c853f165288071b63af9a56b6d76e358fbdcc2
        if (webkitGstCheckVersion(1, 22, 3)) {
            GUniqueOutPtr<char> remoteCertificate;
            GUniqueOutPtr<char> certificate;
            g_object_get(m_backend.get(), "remote-certificate", &remoteCertificate.outPtr(), "certificate", &certificate.outPtr(), nullptr);

            if (remoteCertificate)
                certificates.append(JSC::ArrayBuffer::create(unsafeSpan8(remoteCertificate.get())));

            if (certificate)
                certificates.append(JSC::ArrayBuffer::create(unsafeSpan8(certificate.get())));
        }
        m_client->onStateChanged(toRTCDtlsTransportState(state), WTFMove(certificates));
    });
}

void GStreamerDtlsTransportBackendObserver::start()
{
    g_signal_connect_swapped(m_backend.get(), "notify::state", G_CALLBACK(+[](GStreamerDtlsTransportBackendObserver* observer) {
        observer->stateChanged();
    }), this);
}

void GStreamerDtlsTransportBackendObserver::stop()
{
    m_client = nullptr;
    g_signal_handlers_disconnect_by_data(m_backend.get(), this);
}

GStreamerDtlsTransportBackend::GStreamerDtlsTransportBackend(GRefPtr<GstWebRTCDTLSTransport>&& transport)
    : m_backend(WTFMove(transport))
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_webrtc_dtls_transport_debug, "webkitwebrtcdtls", 0, "WebKit WebRTC DTLS Transport");
    });
    ASSERT(m_backend);
    ASSERT(isMainThread());
}

GStreamerDtlsTransportBackend::~GStreamerDtlsTransportBackend()
{
    unregisterClient();
}

UniqueRef<RTCIceTransportBackend> GStreamerDtlsTransportBackend::iceTransportBackend()
{
    return makeUniqueRef<GStreamerIceTransportBackend>(GRefPtr<GstWebRTCDTLSTransport>(m_backend));
}

void GStreamerDtlsTransportBackend::registerClient(RTCDtlsTransportBackendClient& client)
{
    m_observer = GStreamerDtlsTransportBackendObserver::create(client, GRefPtr<GstWebRTCDTLSTransport>(m_backend));
    m_observer->start();
}

void GStreamerDtlsTransportBackend::unregisterClient()
{
    if (m_observer)
        m_observer->stop();
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
