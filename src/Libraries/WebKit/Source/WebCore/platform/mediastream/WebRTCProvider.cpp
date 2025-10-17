/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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
#include "WebRTCProvider.h"

#include "ContentType.h"
#include "MediaCapabilitiesDecodingInfo.h"
#include "MediaCapabilitiesEncodingInfo.h"
#include "MediaDecodingConfiguration.h"
#include "MediaEncodingConfiguration.h"

#include <wtf/Function.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebRTCProvider);

#if !USE(LIBWEBRTC) && !USE(GSTREAMER_WEBRTC)
UniqueRef<WebRTCProvider> WebRTCProvider::create()
{
    return makeUniqueRef<WebRTCProvider>();
}

bool WebRTCProvider::webRTCAvailable()
{
    return false;
}

void WebRTCProvider::setH264HardwareEncoderAllowed(bool)
{
}

#endif

RefPtr<RTCDataChannelRemoteHandlerConnection> WebRTCProvider::createRTCDataChannelRemoteHandlerConnection()
{
    return nullptr;
}

void WebRTCProvider::setH265Support(bool value)
{
    m_supportsH265 = value;
#if ENABLE(WEB_RTC)
    m_videoDecodingCapabilities = { };
    m_videoEncodingCapabilities = { };
#endif
}

void WebRTCProvider::setVP9Support(bool supportsVP9Profile0, bool supportsVP9Profile2)
{
    m_supportsVP9Profile0 = supportsVP9Profile0;
    m_supportsVP9Profile2 = supportsVP9Profile2;

#if ENABLE(WEB_RTC)
    m_videoDecodingCapabilities = { };
    m_videoEncodingCapabilities = { };
#endif
}

void WebRTCProvider::setAV1Support(bool supportsAV1)
{
    m_supportsAV1 = supportsAV1;

#if ENABLE(WEB_RTC)
    m_videoDecodingCapabilities = { };
    m_videoEncodingCapabilities = { };
#endif
}

bool WebRTCProvider::isSupportingAV1() const
{
    return m_supportsAV1;
}

bool WebRTCProvider::isSupportingH265() const
{
    return m_supportsH265;
}

bool WebRTCProvider::isSupportingVP9Profile0() const
{
    return m_supportsVP9Profile0;
}

bool WebRTCProvider::isSupportingVP9Profile2() const
{
    return m_supportsVP9Profile2;
}

bool WebRTCProvider::isSupportingMDNS() const
{
    return m_supportsMDNS;
}

void WebRTCProvider::setLoggingLevel(WTFLogLevel)
{

}

void WebRTCProvider::clearFactory()
{

}

#if ENABLE(WEB_RTC)

std::optional<RTCRtpCapabilities> WebRTCProvider::receiverCapabilities(const String&)
{
    return { };
}

std::optional<RTCRtpCapabilities> WebRTCProvider::senderCapabilities(const String&)
{
    return { };
}

std::optional<RTCRtpCodecCapability> WebRTCProvider::codecCapability(const ContentType& contentType, const std::optional<RTCRtpCapabilities>& capabilities)
{
    if (!capabilities)
        return { };

    auto containerType = contentType.containerType();
    for (auto& codec : capabilities->codecs) {
        if (equalIgnoringASCIICase(containerType, codec.mimeType))
            return codec;
    }
    return { };
}

std::optional<RTCRtpCapabilities>& WebRTCProvider::audioDecodingCapabilities()
{
    if (!m_audioDecodingCapabilities)
        initializeAudioDecodingCapabilities();
    return m_audioDecodingCapabilities;
}

std::optional<RTCRtpCapabilities>& WebRTCProvider::videoDecodingCapabilities()
{
    if (!m_videoDecodingCapabilities)
        initializeVideoDecodingCapabilities();
    return m_videoDecodingCapabilities;
}

std::optional<RTCRtpCapabilities>& WebRTCProvider::audioEncodingCapabilities()
{
    if (!m_audioEncodingCapabilities)
        initializeAudioEncodingCapabilities();
    return m_audioEncodingCapabilities;
}

std::optional<RTCRtpCapabilities>& WebRTCProvider::videoEncodingCapabilities()
{
    if (!m_videoEncodingCapabilities)
        initializeVideoEncodingCapabilities();
    return m_videoEncodingCapabilities;
}

#endif // ENABLE(WEB_RTC)

std::optional<MediaCapabilitiesInfo> WebRTCProvider::computeVPParameters(const VideoConfiguration&)
{
    return { };
}

bool WebRTCProvider::isVPSoftwareDecoderSmooth(const VideoConfiguration&)
{
    return true;
}

bool WebRTCProvider::isVPXEncoderSmooth(const VideoConfiguration&)
{
    return false;
}

bool WebRTCProvider::isH264EncoderSmooth(const VideoConfiguration&)
{
    return true;
}

void WebRTCProvider::createDecodingConfiguration(MediaDecodingConfiguration&& configuration, DecodingConfigurationCallback&& callback)
{
    ASSERT(configuration.type == MediaDecodingType::WebRTC);

    // FIXME: Validate additional parameters, in particular mime type parameters.
    MediaCapabilitiesDecodingInfo info { WTFMove(configuration) };

#if ENABLE(WEB_RTC)
    if (info.supportedConfiguration.video) {
        ContentType contentType { info.supportedConfiguration.video->contentType };
        auto codec = codecCapability(contentType, videoDecodingCapabilities());
        if (!codec) {
            callback({ });
            return;
        }
        if (auto infoOverride = videoDecodingCapabilitiesOverride(*info.supportedConfiguration.video)) {
            if (!infoOverride->supported) {
                callback({ });
                return;
            }
            info.smooth = infoOverride->smooth;
            info.powerEfficient = infoOverride->powerEfficient;
        }
    }
    if (info.supportedConfiguration.audio) {
        ContentType contentType { info.supportedConfiguration.audio->contentType };
        auto codec = codecCapability(contentType, audioDecodingCapabilities());
        if (!codec) {
            callback({ });
            return;
        }
    }
#endif
    info.supported = true;
    callback(WTFMove(info));
}

void WebRTCProvider::createEncodingConfiguration(MediaEncodingConfiguration&& configuration, EncodingConfigurationCallback&& callback)
{
    ASSERT(configuration.type == MediaEncodingType::WebRTC);

    // FIXME: Validate additional parameters, in particular mime type parameters.
    MediaCapabilitiesEncodingInfo info { WTFMove(configuration) };

#if ENABLE(WEB_RTC)
    if (info.supportedConfiguration.video) {
        ContentType contentType { info.supportedConfiguration.video->contentType };
        auto codec = codecCapability(contentType, videoEncodingCapabilities());
        if (!codec) {
            callback({ });
            return;
        }
        if (auto infoOverride = videoEncodingCapabilitiesOverride(*info.supportedConfiguration.video)) {
            if (!infoOverride->supported) {
                callback({ });
                return;
            }
            info.smooth = infoOverride->smooth;
            info.powerEfficient = infoOverride->powerEfficient;
        }
    }
    if (info.supportedConfiguration.audio) {
        ContentType contentType { info.supportedConfiguration.audio->contentType };
        auto codec = codecCapability(contentType, audioEncodingCapabilities());
        if (!codec) {
            callback({ });
            return;
        }
    }
#endif
    info.supported = true;
    callback(WTFMove(info));
}

void WebRTCProvider::initializeAudioDecodingCapabilities()
{

}

void WebRTCProvider::initializeVideoDecodingCapabilities()
{

}

void WebRTCProvider::initializeAudioEncodingCapabilities()
{

}

void WebRTCProvider::initializeVideoEncodingCapabilities()
{

}

std::optional<MediaCapabilitiesDecodingInfo> WebRTCProvider::videoDecodingCapabilitiesOverride(const VideoConfiguration&)
{
    return { };
}

std::optional<MediaCapabilitiesEncodingInfo> WebRTCProvider::videoEncodingCapabilitiesOverride(const VideoConfiguration&)
{
    return { };
}

void WebRTCProvider::setPortAllocatorRange(StringView range)
{
    if (range.isEmpty())
        return;

    if (range == "0:0"_s)
        return;

    auto components = range.toStringWithoutCopying().split(':');
    if (UNLIKELY(components.size() != 2)) {
        WTFLogAlways("Invalid format for UDP port range. Should be \"min-port:max-port\"");
        ASSERT_NOT_REACHED();
        return;
    }

    auto minPort = WTF::parseInteger<int>(components[0]);
    auto maxPort = WTF::parseInteger<int>(components[1]);
    if (!minPort || !maxPort) {
        WTFLogAlways("Invalid format for UDP port range. Should be \"min-port:max-port\"");
        ASSERT_NOT_REACHED();
        return;
    }

    if (*minPort < 0) {
        WTFLogAlways("Invalid value for UDP minimum port value: %d", *minPort);
        return;
    }

    if (*maxPort < 0) {
        WTFLogAlways("Invalid value for UDP maximum port value: %d", *maxPort);
        return;
    }

    m_portAllocatorRange = { { *minPort, *maxPort } };
}

std::optional<std::pair<int, int>> WebRTCProvider::portAllocatorRange() const
{
    return m_portAllocatorRange;
}

} // namespace WebCore
