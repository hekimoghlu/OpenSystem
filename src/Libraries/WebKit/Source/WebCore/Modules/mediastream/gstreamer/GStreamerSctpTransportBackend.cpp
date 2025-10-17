/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#include "GStreamerSctpTransportBackend.h"

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "GStreamerDtlsTransportBackend.h"
#include "GStreamerWebRTCUtils.h"
#include <wtf/TZoneMallocInlines.h>

GST_DEBUG_CATEGORY(webkit_webrtc_sctp_transport_debug);
#define GST_CAT_DEFAULT webkit_webrtc_sctp_transport_debug

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RTCSctpTransportState);

static inline RTCSctpTransportState toRTCSctpTransportState(GstWebRTCSCTPTransportState state)
{
    switch (state) {
    case GST_WEBRTC_SCTP_TRANSPORT_STATE_NEW:
    case GST_WEBRTC_SCTP_TRANSPORT_STATE_CONNECTING:
        return RTCSctpTransportState::Connecting;
    case GST_WEBRTC_SCTP_TRANSPORT_STATE_CONNECTED:
        return RTCSctpTransportState::Connected;
    case GST_WEBRTC_SCTP_TRANSPORT_STATE_CLOSED:
        return RTCSctpTransportState::Closed;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

GStreamerSctpTransportBackend::GStreamerSctpTransportBackend(GRefPtr<GstWebRTCSCTPTransport>&& transport)
    : m_backend(WTFMove(transport))
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_webrtc_sctp_transport_debug, "webkitwebrtcsctp", 0, "WebKit WebRTC SCTP transport");
    });
    ASSERT(m_backend);
}

GStreamerSctpTransportBackend::~GStreamerSctpTransportBackend()
{
    unregisterClient();
}

UniqueRef<RTCDtlsTransportBackend> GStreamerSctpTransportBackend::dtlsTransportBackend()
{
    GRefPtr<GstWebRTCDTLSTransport> transport;
    g_object_get(m_backend.get(), "transport", &transport.outPtr(), nullptr);
    return makeUniqueRef<GStreamerDtlsTransportBackend>(WTFMove(transport));
}

void GStreamerSctpTransportBackend::registerClient(RTCSctpTransportBackendClient& client)
{
    ASSERT(!m_client);
    m_client = client;
    g_signal_connect_swapped(m_backend.get(), "notify::state", G_CALLBACK(+[](GStreamerSctpTransportBackend* backend) {
        backend->stateChanged();
    }), this);
}

void GStreamerSctpTransportBackend::unregisterClient()
{
    g_signal_handlers_disconnect_by_data(m_backend.get(), this);
    m_client.clear();
}

void GStreamerSctpTransportBackend::stateChanged()
{
    if (!m_client)
        return;

    GstWebRTCSCTPTransportState transportState;
    guint16 maxChannels;
    uint64_t maxMessageSize;
    g_object_get(m_backend.get(), "state", &transportState, "max-message-size", &maxMessageSize, "max-channels", &maxChannels, nullptr);
    GST_DEBUG("Notifying SCTP transport state, max-message-size: %" G_GUINT64_FORMAT " max-channels: %" G_GUINT16_FORMAT, maxMessageSize, maxChannels);
    callOnMainThread([client = m_client, transportState, maxChannels, maxMessageSize] {
        if (!client)
            return;

        client->onStateChanged(toRTCSctpTransportState(transportState), maxMessageSize, maxChannels);
    });
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
