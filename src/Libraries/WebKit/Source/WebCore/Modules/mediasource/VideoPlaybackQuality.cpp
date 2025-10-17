/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#include "VideoPlaybackQuality.h"

#if ENABLE(VIDEO)

#include "MediaPlayer.h"
#include <wtf/JSONValues.h>

namespace WebCore {

Ref<VideoPlaybackQuality> VideoPlaybackQuality::create(double creationTime, const VideoPlaybackQualityMetrics& metrics)
{
    return adoptRef(*new VideoPlaybackQuality(creationTime, metrics));
}

VideoPlaybackQuality::VideoPlaybackQuality(double creationTime, const VideoPlaybackQualityMetrics& metrics)
    : m_creationTime(creationTime)
    , m_totalVideoFrames(metrics.totalVideoFrames)
    , m_droppedVideoFrames(metrics.droppedVideoFrames)
    , m_corruptedVideoFrames(metrics.corruptedVideoFrames)
    , m_displayCompositedVideoFrames(metrics.displayCompositedVideoFrames)
    , m_totalFrameDelay(metrics.totalFrameDelay)
{
}

Ref<JSON::Object> VideoPlaybackQuality::toJSONObject() const
{
    Ref json = JSON::Object::create();

    json->setDouble("creationTime"_s, m_creationTime);
    json->setInteger("totalVideoFrames"_s, m_totalVideoFrames);
    json->setInteger("droppedVideoFrames"_s, m_droppedVideoFrames);
    json->setInteger("corruptedVideoFrames"_s, m_corruptedVideoFrames);
    json->setInteger("displayCompositedVideoFrames"_s, m_displayCompositedVideoFrames);
    json->setDouble("totalFrameDelay"_s, m_totalFrameDelay);

    return json;
}

}

#endif // ENABLE(VIDEO)
