/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

#if ENABLE(GPU_PROCESS)

#include <WebCore/ContentType.h>
#include <WebCore/FourCC.h>
#include <WebCore/LayoutRect.h>
#include <WebCore/PlatformTextTrack.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

struct RemoteMediaPlayerProxyConfiguration {
    String referrer;
    String userAgent;
    String sourceApplicationIdentifier;
    String networkInterfaceName;
    String audioOutputDeviceId;
    Vector<WebCore::ContentType> mediaContentTypesRequiringHardwareSupport;
    std::optional<Vector<String>> allowedMediaContainerTypes;
    std::optional<Vector<String>> allowedMediaCodecTypes;
    std::optional<Vector<WebCore::FourCC>> allowedMediaVideoCodecIDs;
    std::optional<Vector<WebCore::FourCC>> allowedMediaAudioCodecIDs;
    std::optional<Vector<WebCore::FourCC>> allowedMediaCaptionFormatTypes;
    WebCore::LayoutRect playerContentBoxRect;
    Vector<String> preferredAudioCharacteristics;
#if PLATFORM(COCOA)
    Vector<WebCore::PlatformTextTrackData> outOfBandTrackData;
#endif
    WebCore::SecurityOriginData documentSecurityOrigin;
    WebCore::IntSize presentationSize { };
    WebCore::FloatSize videoLayerSize { };
    uint64_t logIdentifier { 0 };
    bool shouldUsePersistentCache { false };
    bool isVideo { false };
    bool renderingCanBeAccelerated { false };
    bool prefersSandboxedParsing { false };
    bool shouldDisableHDR { false };
#if PLATFORM(IOS_FAMILY)
    bool canShowWhileLocked { false };
#endif
};

} // namespace WebKit

#endif
