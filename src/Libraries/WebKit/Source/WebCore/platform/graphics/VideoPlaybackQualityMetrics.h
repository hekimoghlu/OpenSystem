/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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

namespace WebCore {

struct VideoPlaybackQualityMetrics {
    uint32_t totalVideoFrames { 0 };
    uint32_t droppedVideoFrames { 0 };
    uint32_t corruptedVideoFrames { 0 };
    double totalFrameDelay { 0 };
    uint32_t displayCompositedVideoFrames { 0 };

    VideoPlaybackQualityMetrics& operator+=(const VideoPlaybackQualityMetrics& other)
    {
        totalVideoFrames += other.totalVideoFrames;
        droppedVideoFrames += other.droppedVideoFrames;
        corruptedVideoFrames += other.corruptedVideoFrames;
        totalFrameDelay += other.totalFrameDelay;
        displayCompositedVideoFrames += other.displayCompositedVideoFrames;
        return *this;
    }

    VideoPlaybackQualityMetrics isolatedCopy() const { return *this; }
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
