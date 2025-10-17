/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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

// FIXME: We should likely rename this header file to WebRTCProvider.h because depending on the
// build configuration we create either a LibWebRTCProvider, or a GStreamerWebRTCProvider or
// fallback to WebRTCProvider. This rename would open another can of worms though, leading to the
// renaming of more LibWebRTC-prefixed files in WebKit.
// https://bugs.webkit.org/show_bug.cgi?id=243774

#include <wtf/Compiler.h>

#if USE(LIBWEBRTC)

#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

#if PLATFORM(COCOA)
#include <WebCore/LibWebRTCProviderCocoa.h>
#elif USE(GSTREAMER)
#include <WebCore/LibWebRTCProviderGStreamer.h>
#endif
#elif USE(GSTREAMER_WEBRTC)
#include <WebCore/GStreamerWebRTCProvider.h>
#else // !USE(LIBWEBRTC) && !USE(GSTREAMER_WEBRTC)
#include <WebCore/WebRTCProvider.h>
#endif

namespace WebKit {

class WebPage;

#if USE(LIBWEBRTC)

#if PLATFORM(COCOA)
using LibWebRTCProviderBase = WebCore::LibWebRTCProviderCocoa;
#elif USE(GSTREAMER)
using LibWebRTCProviderBase = WebCore::LibWebRTCProviderGStreamer;
#else
using LibWebRTCProviderBase = WebCore::LibWebRTCProvider;
#endif

class LibWebRTCProvider final : public LibWebRTCProviderBase {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCProvider);
public:
    explicit LibWebRTCProvider(WebPage&);
    ~LibWebRTCProvider();

private:
    std::unique_ptr<SuspendableSocketFactory> createSocketFactory(String&& /* userAgent */, WebCore::ScriptExecutionContextIdentifier, bool /* isFirstParty */, WebCore::RegistrableDomain&&) final;

    rtc::scoped_refptr<webrtc::PeerConnectionInterface> createPeerConnection(WebCore::ScriptExecutionContextIdentifier, webrtc::PeerConnectionObserver&, rtc::PacketSocketFactory*, webrtc::PeerConnectionInterface::RTCConfiguration&&) final;

#if PLATFORM(COCOA) && USE(LIBWEBRTC)
    bool isSupportingVP9HardwareDecoder() const final;
    void setVP9HardwareSupportForTesting(std::optional<bool>) final;
#endif
    void disableNonLocalhostConnections() final;
    void startedNetworkThread() final;
    RefPtr<WebCore::RTCDataChannelRemoteHandlerConnection> createRTCDataChannelRemoteHandlerConnection() final;
    void setLoggingLevel(WTFLogLevel) final;

    void willCreatePeerConnectionFactory() final;

    WeakRef<WebPage> m_webPage;
};

inline UniqueRef<LibWebRTCProvider> createLibWebRTCProvider(WebPage& page)
{
    return makeUniqueRef<LibWebRTCProvider>(page);
}

#elif USE(GSTREAMER_WEBRTC)
using LibWebRTCProvider = WebCore::GStreamerWebRTCProvider;

inline UniqueRef<LibWebRTCProvider> createLibWebRTCProvider(WebPage&)
{
    return makeUniqueRef<LibWebRTCProvider>();
}

#else
using LibWebRTCProvider = WebCore::WebRTCProvider;

inline UniqueRef<LibWebRTCProvider> createLibWebRTCProvider(WebPage&)
{
    return makeUniqueRef<LibWebRTCProvider>();
}
#endif // USE(LIBWEBRTC)

} // namespace WebKit
