/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

#if ENABLE(MEDIA_STREAM)

#include "CaptureDevice.h"
#include "MediaStreamTrackHintValue.h"
#include "RealtimeMediaSource.h"
#include <wtf/FastMalloc.h>

namespace WebCore {

class PreventSourceFromEndingObserverWrapper;

struct MediaStreamTrackDataHolder {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    WEBCORE_EXPORT MediaStreamTrackDataHolder(String&& trackId, String&& label, RealtimeMediaSource::Type, CaptureDevice::DeviceType, bool isEnabled, bool isEnded, MediaStreamTrackHintValue, bool isProducingData, bool isMuted, bool isInterrupted, RealtimeMediaSourceSettings, RealtimeMediaSourceCapabilities, Ref<RealtimeMediaSource>&&);
    WEBCORE_EXPORT ~MediaStreamTrackDataHolder();

    MediaStreamTrackDataHolder(const MediaStreamTrackDataHolder &) = delete;
    MediaStreamTrackDataHolder &operator=(const MediaStreamTrackDataHolder &) = delete;

    String trackId;
    String label;
    RealtimeMediaSource::Type type;
    CaptureDevice::DeviceType deviceType;
    bool isEnabled { true };
    bool isEnded { false };
    MediaStreamTrackHintValue contentHint { MediaStreamTrackHintValue::Empty };
    bool isProducingData { false };
    bool isMuted { false };
    bool isInterrupted { false };
    RealtimeMediaSourceSettings settings;
    RealtimeMediaSourceCapabilities capabilities;
    Ref<RealtimeMediaSource> source;

    Ref<PreventSourceFromEndingObserverWrapper> preventSourceFromEndingObserverWrapper;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
