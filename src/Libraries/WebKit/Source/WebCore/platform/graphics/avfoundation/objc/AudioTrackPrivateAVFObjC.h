/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#ifndef AudioTrackPrivateAVFObjC_h
#define AudioTrackPrivateAVFObjC_h

#if ENABLE(VIDEO)

#include "AudioTrackPrivateAVF.h"
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

class AudioTrackPrivateAVFObjC : public AudioTrackPrivateAVF {
    WTF_MAKE_TZONE_ALLOCATED(AudioTrackPrivateAVFObjC);
    WTF_MAKE_NONCOPYABLE(AudioTrackPrivateAVFObjC)
public:
    static RefPtr<AudioTrackPrivateAVFObjC> create(AVPlayerItemTrack* track)
    {
        return adoptRef(new AudioTrackPrivateAVFObjC(track));
    }

    static RefPtr<AudioTrackPrivateAVFObjC> create(AVAssetTrack* track)
    {
        return adoptRef(new AudioTrackPrivateAVFObjC(track));
    }

    static RefPtr<AudioTrackPrivateAVFObjC> create(MediaSelectionOptionAVFObjC& option)
    {
        return adoptRef(new AudioTrackPrivateAVFObjC(option));
    }

    virtual ~AudioTrackPrivateAVFObjC();

    virtual void setEnabled(bool);

    void setPlayerItemTrack(AVPlayerItemTrack*);
    AVPlayerItemTrack* playerItemTrack();

    void setAssetTrack(AVAssetTrack*);
    AVAssetTrack* assetTrack();

    void setMediaSelectionOption(MediaSelectionOptionAVFObjC&);
    MediaSelectionOptionAVFObjC* mediaSelectionOption();

private:
    friend class MediaPlayerPrivateAVFoundationObjC;
    AudioTrackPrivateAVFObjC(AVPlayerItemTrack*);
    AudioTrackPrivateAVFObjC(AVAssetTrack*);
    AudioTrackPrivateAVFObjC(MediaSelectionOptionAVFObjC&);
    AudioTrackPrivateAVFObjC(std::unique_ptr<AVTrackPrivateAVFObjCImpl>&&);

    void resetPropertiesFromTrack();
    void audioTrackConfigurationChanged();
    std::unique_ptr<AVTrackPrivateAVFObjCImpl> m_impl;

    using AudioTrackConfigurationObserver = Observer<void()>;
    AudioTrackConfigurationObserver m_audioTrackConfigurationObserver;
};

}

#endif // ENABLE(VIDEO)


#endif // AudioTrackPrivateAVFObjC_h
