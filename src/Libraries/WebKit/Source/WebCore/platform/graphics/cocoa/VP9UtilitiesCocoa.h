/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#if ENABLE(VP9) && PLATFORM(COCOA)

#include "VP9Utilities.h"
#include <webm/dom_types.h>

namespace vp9_parser {
class Vp9HeaderParser;
}

namespace WebCore {

struct MediaCapabilitiesInfo;
struct VideoConfiguration;
struct VideoInfo;

WEBCORE_EXPORT bool shouldEnableSWVP9Decoder();
WEBCORE_EXPORT void registerWebKitVP9Decoder();
WEBCORE_EXPORT void registerSupplementalVP9Decoder();
bool isVP9DecoderAvailable();
WEBCORE_EXPORT bool vp9HardwareDecoderAvailable();
bool isVP8DecoderAvailable();
bool isVPCodecConfigurationRecordSupported(const VPCodecConfigurationRecord&);
std::optional<MediaCapabilitiesInfo> validateVPParameters(const VPCodecConfigurationRecord&, const VideoConfiguration&);
std::optional<MediaCapabilitiesInfo> computeVPParameters(const VideoConfiguration&, bool vp9HardwareDecoderAvailable);
bool isVPSoftwareDecoderSmooth(const VideoConfiguration&);

Ref<VideoInfo> createVideoInfoFromVP9HeaderParser(const vp9_parser::Vp9HeaderParser&, const webm::Video&);

struct VP8FrameHeader {
    bool keyframe { false };
    uint8_t version { 0 };
    bool showFrame { true };
    uint32_t partitionSize { 0 };
    uint8_t horizontalScale { 0 };
    uint16_t width { 0 };
    uint8_t verticalScale { 0 };
    uint16_t height;
    bool colorSpace { false };
    bool needsClamping { false };
};

std::optional<VP8FrameHeader> parseVP8FrameHeader(std::span<const uint8_t>);
Ref<VideoInfo> createVideoInfoFromVP8Header(const VP8FrameHeader&, const webm::Video&);

class WEBCORE_EXPORT VP9TestingOverrides {
public:
    static VP9TestingOverrides& singleton();

    void setHardwareDecoderDisabled(std::optional<bool>&&);
    std::optional<bool> hardwareDecoderDisabled() const { return m_hardwareDecoderDisabled; }
    
    void setVP9DecoderDisabled(std::optional<bool>&&);
    std::optional<bool> vp9DecoderDisabled() const { return m_vp9DecoderDisabled; }

    void setVP9ScreenSizeAndScale(std::optional<ScreenDataOverrides>&&);
    std::optional<ScreenDataOverrides> vp9ScreenSizeAndScale() const { return m_screenSizeAndScale; }

    void setConfigurationChangedCallback(std::function<void(bool)>&&);
    void resetOverridesToDefaultValues();

    void setSWVPDecodersAlwaysEnabled(bool);
    bool swVPDecodersAlwaysEnabled() const { return m_swVPDecodersAlwaysEnabled; }

private:
    std::optional<bool> m_hardwareDecoderDisabled;
    std::optional<bool> m_vp9DecoderDisabled;
    bool m_swVPDecodersAlwaysEnabled { false };
    std::optional<ScreenDataOverrides> m_screenSizeAndScale;
    Function<void(bool)> m_configurationChangedCallback;
};

}

#endif
