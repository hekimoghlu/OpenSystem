/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#ifndef AVTrackPrivateAVFObjCImpl_h
#define AVTrackPrivateAVFObjCImpl_h

#if ENABLE(VIDEO)

#include "AudioTrackPrivate.h"
#include "PlatformVideoColorSpace.h"
#include "SpatialVideoMetadata.h"
#include "VideoTrackPrivate.h"
#include <wtf/Observer.h>
#include <wtf/Ref.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVAssetTrack;
OBJC_CLASS AVPlayerItem;
OBJC_CLASS AVPlayerItemTrack;
OBJC_CLASS AVMediaSelectionGroup;
OBJC_CLASS AVMediaSelectionOption;

typedef const struct opaqueCMFormatDescription* CMFormatDescriptionRef;

namespace WebCore {
class AVTrackPrivateAVFObjCImpl;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AVTrackPrivateAVFObjCImpl> : std::true_type { };
}

namespace WebCore {

class MediaSelectionOptionAVFObjC;
struct PlatformVideoTrackConfiguration;
struct PlatformAudioTrackConfiguration;

class AVTrackPrivateAVFObjCImpl : public CanMakeWeakPtr<AVTrackPrivateAVFObjCImpl> {
    WTF_MAKE_TZONE_ALLOCATED(AVTrackPrivateAVFObjCImpl);
public:
    explicit AVTrackPrivateAVFObjCImpl(AVPlayerItemTrack*);
    explicit AVTrackPrivateAVFObjCImpl(AVAssetTrack*);
    explicit AVTrackPrivateAVFObjCImpl(MediaSelectionOptionAVFObjC&);
    ~AVTrackPrivateAVFObjCImpl();

    AVPlayerItemTrack* playerItemTrack() const { return m_playerItemTrack.get(); }
    AVAssetTrack* assetTrack() const { return m_assetTrack.get(); }
    MediaSelectionOptionAVFObjC* mediaSelectionOption() const { return m_mediaSelectionOption.get(); }

    bool enabled() const;
    void setEnabled(bool);

    AudioTrackPrivate::Kind audioKind() const;
    VideoTrackPrivate::Kind videoKind() const;

    int index() const;
    TrackID id() const;
    AtomString label() const;
    AtomString language() const;

    static String languageForAVAssetTrack(AVAssetTrack*);
    static String languageForAVMediaSelectionOption(AVMediaSelectionOption *);

    PlatformVideoTrackConfiguration videoTrackConfiguration() const;
    using VideoTrackConfigurationObserver = Observer<void()>;
    void setVideoTrackConfigurationObserver(VideoTrackConfigurationObserver& observer) { m_videoTrackConfigurationObserver = observer; }

    PlatformAudioTrackConfiguration audioTrackConfiguration() const;
    using AudioTrackConfigurationObserver = Observer<void()>;
    void setAudioTrackConfigurationObserver(AudioTrackConfigurationObserver& observer) { m_audioTrackConfigurationObserver = observer; }

private:
    void initializeAssetTrack();

    String codec() const;
    uint32_t width() const;
    uint32_t height() const;
    PlatformVideoColorSpace colorSpace() const;
    double framerate() const;
    uint64_t bitrate() const;
    std::optional<SpatialVideoMetadata> spatialVideoMetadata() const;
    uint32_t sampleRate() const;
    uint32_t numberOfChannels() const;

    RetainPtr<AVPlayerItemTrack> m_playerItemTrack;
    RetainPtr<AVPlayerItem> m_playerItem;
    RefPtr<MediaSelectionOptionAVFObjC> m_mediaSelectionOption;
    RetainPtr<AVAssetTrack> m_assetTrack;
    WeakPtr<VideoTrackConfigurationObserver> m_videoTrackConfigurationObserver;
    WeakPtr<AudioTrackConfigurationObserver> m_audioTrackConfigurationObserver;
};

}

#endif // ENABLE(VIDEO)

#endif
