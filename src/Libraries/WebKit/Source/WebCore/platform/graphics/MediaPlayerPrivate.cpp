/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "MediaPlayerPrivate.h"

#if ENABLE(VIDEO)

#include "VideoFrame.h"
#include "VideoFrameMetadata.h"
#include <wtf/NativePromise.h>

namespace WebCore {

MediaPlayerPrivateInterface::MediaPlayerPrivateInterface() = default;
MediaPlayerPrivateInterface::~MediaPlayerPrivateInterface() = default;

RefPtr<VideoFrame> MediaPlayerPrivateInterface::videoFrameForCurrentTime()
{
    return nullptr;
}

std::optional<VideoFrameMetadata> MediaPlayerPrivateInterface::videoFrameMetadata()
{
    return { };
}

const PlatformTimeRanges& MediaPlayerPrivateInterface::seekable() const
{
    if (maxTimeSeekable() == MediaTime::zeroTime())
        return PlatformTimeRanges::emptyRanges();
    m_seekable = { minTimeSeekable(), maxTimeSeekable() };
    return m_seekable;
}

auto MediaPlayerPrivateInterface::asyncVideoPlaybackQualityMetrics() -> Ref<VideoPlaybackQualityMetricsPromise>
{
    if (auto metrics = videoPlaybackQualityMetrics())
        return VideoPlaybackQualityMetricsPromise::createAndResolve(WTFMove(*metrics));
    return VideoPlaybackQualityMetricsPromise::createAndReject(PlatformMediaError::NotSupportedError);
}

MediaTime MediaPlayerPrivateInterface::currentOrPendingSeekTime() const
{
    auto pendingSeekTime = this->pendingSeekTime();
    if (pendingSeekTime.isValid())
        return pendingSeekTime;
    return currentTime();
}

}

#endif

