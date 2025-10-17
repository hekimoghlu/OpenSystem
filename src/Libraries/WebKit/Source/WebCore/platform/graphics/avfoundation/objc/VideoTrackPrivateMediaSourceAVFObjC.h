/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
#ifndef VideoTrackPrivateMediaSourceAVFObjC_h
#define VideoTrackPrivateMediaSourceAVFObjC_h

#include "IntSize.h"
#include "VideoTrackPrivateAVF.h"
#include <wtf/TZoneMalloc.h>

#if ENABLE(MEDIA_SOURCE)

OBJC_CLASS AVAssetTrack;
OBJC_CLASS AVPlayerItemTrack;

namespace WebCore {

class AVTrackPrivateAVFObjCImpl;
class SourceBufferPrivateAVFObjC;

class VideoTrackPrivateMediaSourceAVFObjC final : public VideoTrackPrivateAVF {
    WTF_MAKE_TZONE_ALLOCATED(VideoTrackPrivateMediaSourceAVFObjC);
    WTF_MAKE_NONCOPYABLE(VideoTrackPrivateMediaSourceAVFObjC)
public:
    static Ref<VideoTrackPrivateMediaSourceAVFObjC> create(AVAssetTrack* track)
    {
        return adoptRef(*new VideoTrackPrivateMediaSourceAVFObjC(track));
    }

    virtual ~VideoTrackPrivateMediaSourceAVFObjC();

    void setAssetTrack(AVAssetTrack*);
    AVAssetTrack* assetTrack() const;

    FloatSize naturalSize() const;

private:
    explicit VideoTrackPrivateMediaSourceAVFObjC(AVAssetTrack*);
    
    void resetPropertiesFromTrack();

    std::unique_ptr<AVTrackPrivateAVFObjCImpl> m_impl;
};

}

#endif // ENABLE(MEDIA_SOURCE)

#endif
