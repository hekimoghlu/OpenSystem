/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#include "MediaEngineConfigurationFactoryMock.h"

#include "ContentType.h"
#include "MediaCapabilitiesDecodingInfo.h"
#include "MediaCapabilitiesEncodingInfo.h"
#include "MediaDecodingConfiguration.h"
#include "MediaEncodingConfiguration.h"

namespace WebCore {

static bool canDecodeMedia(const MediaDecodingConfiguration& configuration)
{
    // The mock implementation supports only local file playback.
    if (configuration.type == MediaDecodingType::MediaSource)
        return false;

    // Maxing out video decoding support at 720P.
    auto videoConfig = configuration.video;
    if (videoConfig && videoConfig->width > 1280 && videoConfig->height > 720)
        return false;

    // Only the "mock-with-alpha" codec supports alphaChannel
    if (videoConfig && videoConfig->alphaChannel && videoConfig->alphaChannel.value()) {
        if (ContentType(videoConfig->contentType).parameter(ContentType::codecsParameter()) != "mock-with-alpha"_s)
            return false;
    }

    // Only the "mock-with-hdr" codec supports HDR)
    if (videoConfig && (videoConfig->colorGamut || videoConfig->hdrMetadataType || videoConfig->transferFunction)) {
        if (ContentType(videoConfig->contentType).parameter(ContentType::codecsParameter()) != "mock-with-hdr"_s)
            return false;
    }

    // Audio decoding support limited to audio/mp4.
    auto audioConfig = configuration.audio;
    if (audioConfig) {
        if (ContentType(audioConfig->contentType).containerType() != "audio/mp4"_s)
            return false;

        // Can only support spatial rendering of tracks with multichannel audio:
        if (audioConfig->spatialRendering.value_or(false) && audioConfig->channels.toDouble() <= 2)
            return false;
    }

    return true;
}

static bool canSmoothlyDecodeMedia(const MediaDecodingConfiguration& configuration)
{
    auto videoConfig = configuration.video;
    if (videoConfig && videoConfig->framerate > 30)
        return false;

    auto audioConfig = configuration.audio;
    if (audioConfig && !audioConfig->channels.isNull())
        return audioConfig->channels == "2"_s;

    return true;
}

static bool canPowerEfficientlyDecodeMedia(const MediaDecodingConfiguration& configuration)
{
    auto videoConfig = configuration.video;
    if (videoConfig && ContentType(videoConfig->contentType).containerType() != "video/mp4"_s)
        return false;

    auto audioConfig = configuration.audio;
    if (audioConfig && audioConfig->bitrate)
        return audioConfig->bitrate.value() <= 1000;

    return true;
}

static bool canEncodeMedia(const MediaEncodingConfiguration& configuration)
{
    ASSERT(configuration.type == MediaEncodingType::Record);
    if (configuration.type != MediaEncodingType::Record)
        return false;

    // Maxing out video encoding support at 720P.
    auto videoConfig = configuration.video;
    if (videoConfig && videoConfig->width > 1280 && videoConfig->height > 720)
        return false;

    // Only the "mock-with-alpha" codec supports alphaChannel
    if (videoConfig && videoConfig->alphaChannel && videoConfig->alphaChannel.value()) {
        if (ContentType(videoConfig->contentType).parameter(ContentType::codecsParameter()) != "mock-with-alpha"_s)
            return false;
    }

    // Audio encoding support limited to audio/mp4.
    auto audioConfig = configuration.audio;
    if (audioConfig && ContentType(audioConfig->contentType).containerType() != "audio/mp4"_s)
        return false;

    return true;
}

static bool canSmoothlyEncodeMedia(const MediaEncodingConfiguration& configuration)
{
    auto videoConfig = configuration.video;
    if (videoConfig && videoConfig->framerate > 30)
        return false;

    auto audioConfig = configuration.audio;
    if (audioConfig && !audioConfig->channels.isNull() && audioConfig->channels != "2"_s)
        return false;

    return true;
}

static bool canPowerEfficientlyEncodeMedia(const MediaEncodingConfiguration& configuration)
{
    auto videoConfig = configuration.video;
    if (videoConfig && ContentType(videoConfig->contentType).containerType() != "video/mp4"_s)
        return false;

    auto audioConfig = configuration.audio;
    if (audioConfig && audioConfig->bitrate && audioConfig->bitrate.value() > 1000)
        return false;

    return true;
}

void MediaEngineConfigurationFactoryMock::createDecodingConfiguration(MediaDecodingConfiguration&& configuration, DecodingConfigurationCallback&& callback)
{
    if (!canDecodeMedia(configuration)) {
        MediaCapabilitiesDecodingInfo info { WTFMove(configuration) };
        callback(WTFMove(info));
        return;
    }
    callback({{ true, canSmoothlyDecodeMedia(configuration), canPowerEfficientlyDecodeMedia(configuration) }, WTFMove(configuration)});
}

void MediaEngineConfigurationFactoryMock::createEncodingConfiguration(MediaEncodingConfiguration&& configuration, EncodingConfigurationCallback&& callback)
{
    if (!canEncodeMedia(configuration)) {
        callback({{ }, WTFMove(configuration) });
        return;
    }
    callback({{ true, canSmoothlyEncodeMedia(configuration), canPowerEfficientlyEncodeMedia(configuration) }, WTFMove(configuration)});
}

} // namespace WebCore
