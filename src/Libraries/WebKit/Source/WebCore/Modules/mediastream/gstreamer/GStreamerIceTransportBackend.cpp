/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#include "GStreamerIceTransportBackend.h"

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "GStreamerWebRTCUtils.h"
#include "NotImplemented.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

GST_DEBUG_CATEGORY(webkit_webrtc_ice_transport_debug);
#define GST_CAT_DEFAULT webkit_webrtc_ice_transport_debug

WTF_MAKE_TZONE_ALLOCATED_IMPL(GStreamerIceTransportBackend);

GStreamerIceTransportBackend::GStreamerIceTransportBackend(GRefPtr<GstWebRTCDTLSTransport>&& transport)
    : m_backend(WTFMove(transport))
{
    ASSERT(m_backend);

    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_webrtc_ice_transport_debug, "webkitwebrtcice", 0, "WebKit WebRTC ICE Transport");
    });

    iceTransportChanged();
    g_signal_connect_swapped(m_backend.get(), "notify::transport", G_CALLBACK(+[](GStreamerIceTransportBackend* backend) {
        backend->iceTransportChanged();
    }), this);
}

GStreamerIceTransportBackend::~GStreamerIceTransportBackend()
{
    if (G_IS_OBJECT(m_iceTransport.get()))
        g_signal_handlers_disconnect_by_data(m_iceTransport.get(), this);
    g_signal_handlers_disconnect_by_data(m_backend.get(), this);
}

void GStreamerIceTransportBackend::iceTransportChanged()
{
    if (G_IS_OBJECT(m_iceTransport.get()))
        g_signal_handlers_disconnect_by_data(m_iceTransport.get(), this);

    g_object_get(m_backend.get(), "transport", &m_iceTransport.outPtr(), nullptr);

    g_signal_connect_swapped(m_iceTransport.get(), "notify::state", G_CALLBACK(+[](GStreamerIceTransportBackend* backend) {
        backend->stateChanged();
    }), this);
    g_signal_connect_swapped(m_iceTransport.get(), "notify::gathering-state", G_CALLBACK(+[](GStreamerIceTransportBackend* backend) {
        backend->gatheringStateChanged();
    }), this);
    g_signal_connect_swapped(m_iceTransport.get(), "on-selected-candidate-pair-change", G_CALLBACK(+[](GStreamerIceTransportBackend* backend) {
        backend->selectedCandidatePairChanged();
    }), this);
}

void GStreamerIceTransportBackend::registerClient(RTCIceTransportBackendClient& client)
{
    ASSERT(!m_client);
    m_client = client;

    GstWebRTCICEConnectionState transportState;
    GstWebRTCICEGatheringState gatheringState;
    g_object_get(m_iceTransport.get(), "state", &transportState, "gathering-state", &gatheringState, nullptr);

    callOnMainThread([weakThis = WeakPtr { *this }, transportState, gatheringState] {
        if (!weakThis || !weakThis->m_client)
            return;

#ifndef GST_DISABLE_GST_DEBUG
        GUniquePtr<char> desc(g_enum_to_string(GST_TYPE_WEBRTC_ICE_CONNECTION_STATE, transportState));
        GST_DEBUG_OBJECT(weakThis->m_backend.get(), "Initial ICE transport state: %s", desc.get());
#endif

        // We start observing a bit late and might miss the checking state. Synthesize it as needed.
        if (transportState > GST_WEBRTC_ICE_CONNECTION_STATE_CHECKING && transportState != GST_WEBRTC_ICE_CONNECTION_STATE_CLOSED)
            weakThis->m_client->onStateChanged(RTCIceTransportState::Checking);

        weakThis->m_client->onStateChanged(toRTCIceTransportState(transportState));
        weakThis->m_client->onGatheringStateChanged(toRTCIceGatheringState(gatheringState));
    });
}

void GStreamerIceTransportBackend::unregisterClient()
{
    ASSERT(m_client);
    m_client.clear();
}

void GStreamerIceTransportBackend::stateChanged() const
{
    if (!m_client)
        return;

    GstWebRTCICEConnectionState transportState;
    g_object_get(m_iceTransport.get(), "state", &transportState, nullptr);

#ifndef GST_DISABLE_GST_DEBUG
    GUniquePtr<char> desc(g_enum_to_string(GST_TYPE_WEBRTC_ICE_CONNECTION_STATE, transportState));
    GST_DEBUG_OBJECT(m_backend.get(), "ICE transport state changed to %s", desc.get());
#endif

    callOnMainThread([weakThis = WeakPtr { *this }, transportState] {
        if (!weakThis || !weakThis->m_client)
            return;
        weakThis->m_client->onStateChanged(toRTCIceTransportState(transportState));
    });
}

void GStreamerIceTransportBackend::gatheringStateChanged() const
{
    if (!m_client)
        return;

    GstWebRTCICEGatheringState gatheringState;
    g_object_get(m_iceTransport.get(), "gathering-state", &gatheringState, nullptr);
    callOnMainThread([weakThis = WeakPtr { *this }, gatheringState] {
        if (!weakThis || !weakThis->m_client)
            return;
        weakThis->m_client->onGatheringStateChanged(toRTCIceGatheringState(gatheringState));
    });
}

void GStreamerIceTransportBackend::selectedCandidatePairChanged()
{
    // FIXME: call m_client->onSelectedCandidatePairChanged(). See also
    // https://github.com/WebKit/WebKit/commit/0692fae10c8e53deba214fd080a35f7c54bd6985
    notImplemented();
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
