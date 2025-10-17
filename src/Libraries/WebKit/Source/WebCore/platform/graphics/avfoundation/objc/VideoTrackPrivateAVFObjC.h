/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#ifndef VideoTrackPrivateAVFObjC_h
#define VideoTrackPrivateAVFObjC_h

#if ENABLE(VIDEO)

#include "VideoTrackPrivateAVF.h"
#include <wtf/Observer.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVAssetTrack;
OBJC_CLASS AVPlayerItem;
OBJC_CLASS AVPlayerItemTrack;
OBJC_CLASS AVMediaSelectionGroup;
OBJC_CLASS AVMediaSelectionOption;

namespace WebCore {

class AVTrackPrivateAVFObjCImpl;
class MediaSelectionOptionAVFObjC;

class VideoTrackPrivateAVFObjC final : public VideoTrackPrivateAVF {
    WTF_MAKE_TZONE_ALLOCATED(VideoTrackPrivateAVFObjC);
    WTF_MAKE_NONCOPYABLE(VideoTrackPrivateAVFObjC)
public:
    static RefPtr<VideoTrackPrivateAVFObjC> create(AVPlayerItemTrack* track)
    {
        return adoptRef(new VideoTrackPrivateAVFObjC(track));
    }

    static RefPtr<VideoTrackPrivateAVFObjC> create(AVAssetTrack* track)
    {
        return adoptRef(new VideoTrackPrivateAVFObjC(track));
    }

    static RefPtr<VideoTrackPrivateAVFObjC> create(MediaSelectionOptionAVFObjC& option)
    {
        return adoptRef(new VideoTrackPrivateAVFObjC(option));
    }

    void setSelected(bool) override;

    void setPlayerItemTrack(AVPlayerItemTrack*);
    AVPlayerItemTrack* playerItemTrack();

    void setAssetTrack(AVAssetTrack*);
    AVAssetTrack* assetTrack();

    void setMediaSelectonOption(MediaSelectionOptionAVFObjC&);
    MediaSelectionOptionAVFObjC* mediaSelectionOption();

private:
    friend class MediaPlayerPrivateAVFoundationObjC;
    explicit VideoTrackPrivateAVFObjC(AVPlayerItemTrack*);
    explicit VideoTrackPrivateAVFObjC(AVAssetTrack*);
    explicit VideoTrackPrivateAVFObjC(MediaSelectionOptionAVFObjC&);
    explicit VideoTrackPrivateAVFObjC(std::unique_ptr<AVTrackPrivateAVFObjCImpl>&&);

    void resetPropertiesFromTrack();
    void videoTrackConfigurationChanged();
    std::unique_ptr<AVTrackPrivateAVFObjCImpl> m_impl;

    using VideoTrackConfigurationObserver = Observer<void()>;
    VideoTrackConfigurationObserver m_videoTrackConfigurationObserver;
};

}

#endif // ENABLE(VIDEO)

#endif // VideoTrackPrivateAVFObjC_h
