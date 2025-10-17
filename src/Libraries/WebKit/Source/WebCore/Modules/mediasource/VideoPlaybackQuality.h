/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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

#if ENABLE(VIDEO)

#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

struct VideoPlaybackQualityMetrics;

class VideoPlaybackQuality : public RefCounted<VideoPlaybackQuality> {
    WTF_MAKE_NONCOPYABLE(VideoPlaybackQuality)
public:
    static Ref<VideoPlaybackQuality> create(double creationTime, const VideoPlaybackQualityMetrics&);

    double creationTime() const { return m_creationTime; }
    unsigned totalVideoFrames() const { return m_totalVideoFrames; }
    unsigned droppedVideoFrames() const { return m_droppedVideoFrames; }
    unsigned corruptedVideoFrames() const { return m_corruptedVideoFrames; }
    unsigned displayCompositedVideoFrames() const { return m_displayCompositedVideoFrames; }
    double totalFrameDelay() const { return m_totalFrameDelay; }

    Ref<JSON::Object> toJSONObject() const;

private:
    VideoPlaybackQuality(double creationTime, const VideoPlaybackQualityMetrics&);

    double m_creationTime;
    uint32_t m_totalVideoFrames;
    uint32_t m_droppedVideoFrames;
    uint32_t m_corruptedVideoFrames;
    uint32_t m_displayCompositedVideoFrames;
    double m_totalFrameDelay;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
