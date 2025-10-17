/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#include "LibWebRTCProviderCocoa.h"

#if USE(LIBWEBRTC)

#include "DeprecatedGlobalSettings.h"
#include "MediaCapabilitiesInfo.h"
#include "VP9UtilitiesCocoa.h"

#include <webrtc/api/create_peerconnection_factory.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/webkit_sdk/WebKit/WebKitDecoder.h>
#include <webrtc/webkit_sdk/WebKit/WebKitEncoder.h>
#include <webrtc/webkit_sdk/WebKit/WebKitVP9Decoder.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/darwin/WeakLinking.h>

WTF_WEAK_LINK_FORCE_IMPORT(webrtc::CreatePeerConnectionFactory);

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCProviderCocoa);

UniqueRef<WebRTCProvider> WebRTCProvider::create()
{
    return makeUniqueRef<LibWebRTCProviderCocoa>();
}

void WebRTCProvider::setH264HardwareEncoderAllowed(bool allowed)
{
    if (webRTCAvailable())
        webrtc::setH264HardwareEncoderAllowed(allowed);
}

LibWebRTCProviderCocoa::~LibWebRTCProviderCocoa()
{
}

std::unique_ptr<webrtc::VideoDecoderFactory> LibWebRTCProviderCocoa::createDecoderFactory()
{
    ASSERT(isMainThread());

    if (!webRTCAvailable())
        return nullptr;

    auto vp9Support = isSupportingVP9Profile2() ? webrtc::WebKitVP9::Profile0And2 : isSupportingVP9Profile0() ? webrtc::WebKitVP9::Profile0 : webrtc::WebKitVP9::Off;
    return webrtc::createWebKitDecoderFactory(isSupportingH265() ? webrtc::WebKitH265::On : webrtc::WebKitH265::Off, vp9Support, vp9HardwareDecoderAvailable() ? webrtc::WebKitVP9VTB::On : webrtc::WebKitVP9VTB::Off, isSupportingAV1() ? webrtc::WebKitAv1::On : webrtc::WebKitAv1::Off);
}

std::unique_ptr<webrtc::VideoEncoderFactory> LibWebRTCProviderCocoa::createEncoderFactory()
{
    ASSERT(isMainThread());

    if (!webRTCAvailable())
        return nullptr;

    auto vp9Support = isSupportingVP9Profile2() ? webrtc::WebKitVP9::Profile0And2 : isSupportingVP9Profile0() ? webrtc::WebKitVP9::Profile0 : webrtc::WebKitVP9::Off;
    return webrtc::createWebKitEncoderFactory(isSupportingH265() ? webrtc::WebKitH265::On : webrtc::WebKitH265::Off, vp9Support, isSupportingAV1() ? webrtc::WebKitAv1::On : webrtc::WebKitAv1::Off);
}

std::optional<MediaCapabilitiesInfo> LibWebRTCProviderCocoa::computeVPParameters(const VideoConfiguration& configuration)
{
    return WebCore::computeVPParameters(configuration, isSupportingVP9HardwareDecoder());
}

bool LibWebRTCProviderCocoa::isVPSoftwareDecoderSmooth(const VideoConfiguration& configuration)
{
    return WebCore::isVPSoftwareDecoderSmooth(configuration);
}

bool WebRTCProvider::webRTCAvailable()
{
#if PLATFORM(IOS) || PLATFORM(VISION)
    ASSERT_WITH_MESSAGE(!!webrtc::CreatePeerConnectionFactory, "Failed to find or load libwebrtc");
    return true;
#else
    return !!webrtc::CreatePeerConnectionFactory;
#endif
}

void LibWebRTCProvider::registerWebKitVP9Decoder()
{
    if (webRTCAvailable())
        webrtc::registerWebKitVP9Decoder();
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
