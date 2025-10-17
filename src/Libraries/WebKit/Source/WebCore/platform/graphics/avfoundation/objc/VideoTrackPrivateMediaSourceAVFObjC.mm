/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#import "config.h"
#import "VideoTrackPrivateMediaSourceAVFObjC.h"

#if ENABLE(MEDIA_SOURCE)

#import "AVTrackPrivateAVFObjCImpl.h"
#import "SourceBufferPrivateAVFObjC.h"
#import <AVFoundation/AVAssetTrack.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoTrackPrivateMediaSourceAVFObjC);

VideoTrackPrivateMediaSourceAVFObjC::VideoTrackPrivateMediaSourceAVFObjC(AVAssetTrack* track)
    : m_impl(makeUnique<AVTrackPrivateAVFObjCImpl>(track))
{
    resetPropertiesFromTrack();
}

VideoTrackPrivateMediaSourceAVFObjC::~VideoTrackPrivateMediaSourceAVFObjC() = default;

void VideoTrackPrivateMediaSourceAVFObjC::resetPropertiesFromTrack()
{
    setTrackIndex(m_impl->index());
    setKind(m_impl->videoKind());
    setId(m_impl->id());
    setLabel(m_impl->label());
    setLanguage(m_impl->language());
    setConfiguration(m_impl->videoTrackConfiguration());
}

void VideoTrackPrivateMediaSourceAVFObjC::setAssetTrack(AVAssetTrack *track)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(track);
    resetPropertiesFromTrack();
}

AVAssetTrack* VideoTrackPrivateMediaSourceAVFObjC::assetTrack() const
{
    return m_impl->assetTrack();
}

FloatSize VideoTrackPrivateMediaSourceAVFObjC::naturalSize() const
{
    return FloatSize([assetTrack() naturalSize]);
}

}

#endif
