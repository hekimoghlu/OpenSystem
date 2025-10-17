/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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

#include "DoubleRange.h"
#include "LongRange.h"
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RealtimeMediaSourceCapabilities;

struct MediaTrackCapabilities {
    std::optional<LongRange> width;
    std::optional<LongRange> height;
    std::optional<DoubleRange> aspectRatio;
    std::optional<DoubleRange> frameRate;
    std::optional<Vector<String>> facingMode;
    std::optional<DoubleRange> volume;
    std::optional<LongRange> sampleRate;
    std::optional<LongRange> sampleSize;
    std::optional<Vector<bool>> echoCancellation;
    String deviceId;
    String groupId;
    String displaySurface;
    std::optional<DoubleRange> focusDistance;
    std::optional<Vector<String>> whiteBalanceMode;
    std::optional<DoubleRange> zoom;
    std::optional<bool> torch;
    std::optional<Vector<bool>> backgroundBlur;
    std::optional<Vector<bool>> powerEfficient;
};

MediaTrackCapabilities toMediaTrackCapabilities(const RealtimeMediaSourceCapabilities&);
} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
