/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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

#include "MDNSRegisterError.h"
#include "MediaCapabilitiesInfo.h"
#include "RTCDataChannelRemoteHandlerConnection.h"
#include "RTCRtpCapabilities.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Expected.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContentType;
struct VideoConfiguration;
struct MediaCapabilitiesDecodingInfo;
struct MediaCapabilitiesEncodingInfo;
struct MediaDecodingConfiguration;
struct MediaEncodingConfiguration;

class WEBCORE_EXPORT WebRTCProvider {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WebRTCProvider, WEBCORE_EXPORT);
public:
    static UniqueRef<WebRTCProvider> create();
    WebRTCProvider() = default;
    virtual ~WebRTCProvider() = default;

    static bool webRTCAvailable();
    static void setH264HardwareEncoderAllowed(bool);

    virtual RefPtr<RTCDataChannelRemoteHandlerConnection> createRTCDataChannelRemoteHandlerConnection();

    using DecodingConfigurationCallback = Function<void(MediaCapabilitiesDecodingInfo&&)>;
    using EncodingConfigurationCallback = Function<void(MediaCapabilitiesEncodingInfo&&)>;
    void createDecodingConfiguration(MediaDecodingConfiguration&&, DecodingConfigurationCallback&&);
    void createEncodingConfiguration(MediaEncodingConfiguration&&, EncodingConfigurationCallback&&);

    void setAV1Support(bool);
    void setH265Support(bool);
    void setVP9Support(bool supportsVP9Profile0, bool supportsVP9Profile2);
    bool isSupportingAV1() const;
    bool isSupportingH265() const;
    bool isSupportingVP9Profile0() const;
    bool isSupportingVP9Profile2() const;

    bool isSupportingMDNS() const;

#if ENABLE(WEB_RTC)
    virtual std::optional<RTCRtpCapabilities> receiverCapabilities(const String&);
    virtual std::optional<RTCRtpCapabilities> senderCapabilities(const String&);
#endif

    virtual void setLoggingLevel(WTFLogLevel);
    virtual void clearFactory();

    void setPortAllocatorRange(StringView);
    std::optional<std::pair<int, int>> portAllocatorRange() const;

protected:
#if ENABLE(WEB_RTC)
    std::optional<RTCRtpCapabilities>& audioDecodingCapabilities();
    std::optional<RTCRtpCapabilities>& videoDecodingCapabilities();
    std::optional<RTCRtpCapabilities>& audioEncodingCapabilities();
    std::optional<RTCRtpCapabilities>& videoEncodingCapabilities();

    std::optional<RTCRtpCodecCapability> codecCapability(const ContentType&, const std::optional<RTCRtpCapabilities>&);

    std::optional<RTCRtpCapabilities> m_audioDecodingCapabilities;
    std::optional<RTCRtpCapabilities> m_videoDecodingCapabilities;
    std::optional<RTCRtpCapabilities> m_audioEncodingCapabilities;
    std::optional<RTCRtpCapabilities> m_videoEncodingCapabilities;
#endif

    virtual std::optional<MediaCapabilitiesInfo> computeVPParameters(const VideoConfiguration&);
    virtual bool isVPSoftwareDecoderSmooth(const VideoConfiguration&);
    virtual bool isVPXEncoderSmooth(const VideoConfiguration&);
    virtual bool isH264EncoderSmooth(const VideoConfiguration&);

    bool m_supportsAV1 { false };
    bool m_supportsH265 { false };
    bool m_supportsVP9Profile0 { false };
    bool m_supportsVP9Profile2 { false };
    bool m_supportsMDNS { false };

    std::optional<std::pair<int, int>> m_portAllocatorRange;

private:
    virtual void initializeAudioDecodingCapabilities();
    virtual void initializeVideoDecodingCapabilities();
    virtual void initializeAudioEncodingCapabilities();
    virtual void initializeVideoEncodingCapabilities();

    virtual std::optional<MediaCapabilitiesDecodingInfo> videoDecodingCapabilitiesOverride(const VideoConfiguration&);
    virtual std::optional<MediaCapabilitiesEncodingInfo> videoEncodingCapabilitiesOverride(const VideoConfiguration&);
};

} // namespace WebCore
