/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#if USE(GSTREAMER_WEBRTC)
#include "GStreamerWebRTCProvider.h"

#include "ContentType.h"
#include "GStreamerRegistryScanner.h"
#include "MediaCapabilitiesDecodingInfo.h"
#include "MediaCapabilitiesEncodingInfo.h"
#include "MediaDecodingConfiguration.h"
#include "MediaEncodingConfiguration.h"
#include "NotImplemented.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GStreamerWebRTCProvider);

void WebRTCProvider::setH264HardwareEncoderAllowed(bool)
{
    // TODO: Hook this into GStreamerRegistryScanner.
    notImplemented();
}

UniqueRef<WebRTCProvider> WebRTCProvider::create()
{
    return makeUniqueRef<GStreamerWebRTCProvider>();
}

bool WebRTCProvider::webRTCAvailable()
{
    return true;
}

std::optional<RTCRtpCapabilities> GStreamerWebRTCProvider::receiverCapabilities(const String& kind)
{
    if (kind == "audio"_s)
        return audioDecodingCapabilities();
    if (kind == "video"_s)
        return videoDecodingCapabilities();

    return { };
}

std::optional<RTCRtpCapabilities> GStreamerWebRTCProvider::senderCapabilities(const String& kind)
{
    if (kind == "audio"_s)
        return audioEncodingCapabilities();
    if (kind == "video"_s)
        return videoEncodingCapabilities();
    return { };
}

void GStreamerWebRTCProvider::initializeAudioEncodingCapabilities()
{
    m_audioEncodingCapabilities = GStreamerRegistryScanner::singleton().audioRtpCapabilities(GStreamerRegistryScanner::Configuration::Encoding);
}

void GStreamerWebRTCProvider::initializeVideoEncodingCapabilities()
{
    ensureGStreamerInitialized();
    registerWebKitGStreamerVideoEncoder();
    m_videoEncodingCapabilities = GStreamerRegistryScanner::singleton().videoRtpCapabilities(GStreamerRegistryScanner::Configuration::Encoding);
    m_videoEncodingCapabilities->codecs.removeAllMatching([isSupportingVP9Profile0 = isSupportingVP9Profile0(), isSupportingVP9Profile2 = isSupportingVP9Profile2(), isSupportingH265 = isSupportingH265()](const auto& codec) {
        if (!isSupportingVP9Profile0 && codec.sdpFmtpLine == "profile-id=0"_s)
            return true;
        if (!isSupportingVP9Profile2 && codec.sdpFmtpLine == "profile-id=2"_s)
            return true;
        if (!isSupportingH265 && codec.mimeType == "video/H265"_s)
            return true;

        return false;
    });
}

void GStreamerWebRTCProvider::initializeAudioDecodingCapabilities()
{
    m_audioDecodingCapabilities = GStreamerRegistryScanner::singleton().audioRtpCapabilities(GStreamerRegistryScanner::Configuration::Decoding);
}

void GStreamerWebRTCProvider::initializeVideoDecodingCapabilities()
{
    m_videoDecodingCapabilities = GStreamerRegistryScanner::singleton().videoRtpCapabilities(GStreamerRegistryScanner::Configuration::Decoding);
    m_videoDecodingCapabilities->codecs.removeAllMatching([isSupportingVP9Profile0 = isSupportingVP9Profile0(), isSupportingVP9Profile2 = isSupportingVP9Profile2(), isSupportingH265 = isSupportingH265()](const auto& codec) {
        if (!isSupportingVP9Profile0 && codec.sdpFmtpLine == "profile-id=0"_s)
            return true;
        if (!isSupportingVP9Profile2 && codec.sdpFmtpLine == "profile-id=2"_s)
            return true;
        if (!isSupportingH265 && codec.mimeType == "video/H265"_s)
            return true;

        return false;
    });
}

std::optional<MediaCapabilitiesDecodingInfo> GStreamerWebRTCProvider::videoDecodingCapabilitiesOverride(const VideoConfiguration& configuration)
{
    MediaCapabilitiesDecodingInfo info;
    info.supportedConfiguration.type = MediaDecodingType::WebRTC;
    ContentType contentType { configuration.contentType };
    auto containerType = contentType.containerType();
    if (equalLettersIgnoringASCIICase(containerType, "video/vp8"_s)) {
        info.powerEfficient = false;
        info.smooth = isVPSoftwareDecoderSmooth(configuration);
    } else if (equalLettersIgnoringASCIICase(containerType, "video/vp9"_s)) {
        auto decodingInfo = computeVPParameters(configuration);
        info.powerEfficient = decodingInfo ? decodingInfo->powerEfficient : true;
        info.smooth = decodingInfo ? decodingInfo->smooth : isVPSoftwareDecoderSmooth(configuration);
    } else {
        // FIXME: Provide more granular H.264 decoder information.
        info.powerEfficient = true;
        info.smooth = true;
    }
    info.supported = true;
    return { info };
}

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
