/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
#include "MediaEngineConfigurationFactoryCocoa.h"

#if PLATFORM(COCOA)

#include "AV1Utilities.h"
#include "AV1UtilitiesCocoa.h"
#include "HEVCUtilitiesCocoa.h"
#include "MediaCapabilitiesDecodingInfo.h"
#include "MediaDecodingConfiguration.h"
#include "MediaPlayer.h"
#include "MediaSessionHelperIOS.h"
#include "PlatformMediaSessionManager.h"
#include "VP9UtilitiesCocoa.h"
#include <pal/avfoundation/OutputContext.h>
#include <pal/avfoundation/OutputDevice.h>
#include <wtf/Algorithms.h>

#include "VideoToolboxSoftLink.h"
#include <pal/cf/AudioToolboxSoftLink.h>

namespace WebCore {

static CMVideoCodecType videoCodecTypeFromRFC4281Type(StringView type)
{
    if (type.startsWith("mp4v"_s))
        return kCMVideoCodecType_MPEG4Video;
    if (type.startsWith("avc1"_s) || type.startsWith("avc3"_s))
        return kCMVideoCodecType_H264;
    if (type.startsWith("hvc1"_s) || type.startsWith("hev1"_s))
        return kCMVideoCodecType_HEVC;
#if ENABLE(VP9)
    if (type.startsWith("vp09"_s))
        return kCMVideoCodecType_VP9;
#endif
    return 0;
}

static std::optional<MediaCapabilitiesInfo> computeMediaCapabilitiesInfo(const MediaDecodingConfiguration& configuration)
{
    MediaCapabilitiesInfo info;

    if (configuration.video) {
        auto& videoConfiguration = configuration.video.value();
        MediaEngineSupportParameters parameters { };
        parameters.allowedMediaContainerTypes = configuration.allowedMediaContainerTypes;
        parameters.allowedMediaCodecTypes = configuration.allowedMediaCodecTypes;

        switch (configuration.type) {
        case MediaDecodingType::File:
            parameters.isMediaSource = false;
            break;
        case MediaDecodingType::MediaSource:
            parameters.isMediaSource = true;
            break;
        case MediaDecodingType::WebRTC:
            ASSERT_NOT_REACHED();
            return std::nullopt;
        }

        parameters.type = ContentType(videoConfiguration.contentType);
        if (MediaPlayer::supportsType(parameters) != MediaPlayer::SupportsType::IsSupported)
            return std::nullopt;

        auto codecs = parameters.type.codecs();
        if (codecs.size() != 1)
            return std::nullopt;

        info.supported = true;
        auto& codec = codecs[0];
        auto videoCodecType = videoCodecTypeFromRFC4281Type(codec);

        bool hdrSupported = videoConfiguration.colorGamut || videoConfiguration.hdrMetadataType || videoConfiguration.transferFunction;
        bool alphaChannel = videoConfiguration.alphaChannel && videoConfiguration.alphaChannel.value();

        if (videoCodecType == kCMVideoCodecType_HEVC) {
            auto parameters = parseHEVCCodecParameters(codec);
            if (!parameters)
                return std::nullopt;
            auto parsedInfo = validateHEVCParameters(*parameters, alphaChannel, hdrSupported);
            if (!parsedInfo)
                return std::nullopt;
            info = *parsedInfo;
        } else if (codec.startsWith("dvh1"_s) || codec.startsWith("dvhe"_s)) {
            auto parameters = parseDoViCodecParameters(codec);
            if (!parameters)
                return std::nullopt;
            auto parsedInfo = validateDoViParameters(*parameters, alphaChannel, hdrSupported);
            if (!parsedInfo)
                return std::nullopt;
            info = *parsedInfo;
#if ENABLE(VP9)
        } else if (videoCodecType == kCMVideoCodecType_VP9) {
            if (!configuration.canExposeVP9)
                return std::nullopt;
            auto parameters = parseVPCodecParameters(codec);
            if (!parameters)
                return std::nullopt;
            auto parsedInfo = validateVPParameters(*parameters, videoConfiguration);
            if (!parsedInfo)
                return std::nullopt;
            info = *parsedInfo;
#endif
#if ENABLE(AV1)
        } else if (codec.startsWith("av01"_s)) {
            auto parameters = parseAV1CodecParameters(codec);
            if (!parameters)
                return std::nullopt;
            auto parsedInfo = validateAV1Parameters(*parameters, videoConfiguration);
            if (!parsedInfo)
                return std::nullopt;
            info = *parsedInfo;
#endif
        } else if (videoCodecType) {
            if (alphaChannel || hdrSupported)
                return std::nullopt;

            if (canLoad_VideoToolbox_VTIsHardwareDecodeSupported()) {
                info.powerEfficient = VTIsHardwareDecodeSupported(videoCodecType);
                info.smooth = true;
            }
        } else
            return std::nullopt;
    }

    if (!configuration.audio)
        return info;

    MediaEngineSupportParameters parameters { };
    parameters.type = ContentType(configuration.audio.value().contentType);
    parameters.isMediaSource = configuration.type == MediaDecodingType::MediaSource;
    parameters.allowedMediaContainerTypes = configuration.allowedMediaContainerTypes;
    parameters.allowedMediaCodecTypes = configuration.allowedMediaCodecTypes;

    if (MediaPlayer::supportsType(parameters) != MediaPlayer::SupportsType::IsSupported)
        return std::nullopt;

    info.supported = true;
    if (!configuration.audio->spatialRendering.value_or(false))
        return info;

    auto supportsSpatialPlayback = PlatformMediaSessionManager::singleton().supportsSpatialAudioPlaybackForConfiguration(configuration);
    if (!supportsSpatialPlayback.has_value())
        return std::nullopt;

    info.supported = supportsSpatialPlayback.value();

    return info;
}

void createMediaPlayerDecodingConfigurationCocoa(MediaDecodingConfiguration&& configuration, Function<void(MediaCapabilitiesDecodingInfo&&)>&& callback)
{
    auto info = computeMediaCapabilitiesInfo(configuration);
    if (!info)
        callback({ { }, WTFMove(configuration) });
    else {
        MediaCapabilitiesDecodingInfo infoWithConfiguration = { WTFMove(*info), WTFMove(configuration) };
        callback(WTFMove(infoWithConfiguration));
    }
}

}
#endif
