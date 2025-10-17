/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#include "MediaEngineConfigurationFactoryGStreamer.h"

#if USE(GSTREAMER)

#include "GStreamerRegistryScanner.h"
#include "MediaCapabilitiesDecodingInfo.h"
#include "MediaCapabilitiesEncodingInfo.h"
#include "MediaDecodingConfiguration.h"
#include "MediaEncodingConfiguration.h"
#include "MediaPlayer.h"
#include <wtf/Function.h>

#if ENABLE(MEDIA_SOURCE)
#include "GStreamerRegistryScannerMSE.h"
#endif

namespace WebCore {

void createMediaPlayerDecodingConfigurationGStreamer(MediaDecodingConfiguration&& configuration, Function<void(MediaCapabilitiesDecodingInfo&&)>&& callback)
{
    bool isMediaSource = configuration.type == MediaDecodingType::MediaSource;
#if ENABLE(MEDIA_SOURCE)
    auto& scanner = isMediaSource ? GStreamerRegistryScannerMSE::singleton() : GStreamerRegistryScanner::singleton();
#else
    if (isMediaSource) {
        callback({{ }, WTFMove(configuration)});
        return;
    }
    auto& scanner = GStreamerRegistryScanner::singleton();
#endif
    auto lookupResult = scanner.isDecodingSupported(configuration);
    MediaCapabilitiesDecodingInfo info;
    info.supported = lookupResult.isSupported;
    info.powerEfficient = lookupResult.isUsingHardware;
    info.supportedConfiguration = WTFMove(configuration);

    callback(WTFMove(info));
}

void createMediaPlayerEncodingConfigurationGStreamer(MediaEncodingConfiguration&& configuration, Function<void(MediaCapabilitiesEncodingInfo&&)>&& callback)
{
    auto& scanner = GStreamerRegistryScanner::singleton();
    auto lookupResult = scanner.isEncodingSupported(configuration);
    MediaCapabilitiesEncodingInfo info;
    info.supported = lookupResult.isSupported;
    info.powerEfficient = lookupResult.isUsingHardware;
    info.supportedConfiguration = WTFMove(configuration);

    callback(WTFMove(info));
}
}
#endif
