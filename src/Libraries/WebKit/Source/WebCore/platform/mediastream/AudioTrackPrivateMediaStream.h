/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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

#include "AudioTrackPrivate.h"
#include "MediaStreamTrackPrivate.h"
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AudioMediaStreamTrackRenderer;

class AudioTrackPrivateMediaStream final
    : public AudioTrackPrivate
    , public MediaStreamTrackPrivateObserver
    , private RealtimeMediaSource::AudioSampleObserver
    , public CanMakeCheckedPtr<AudioTrackPrivateMediaStream> {
    WTF_MAKE_TZONE_ALLOCATED(AudioTrackPrivateMediaStream);
    WTF_MAKE_NONCOPYABLE(AudioTrackPrivateMediaStream)
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(AudioTrackPrivateMediaStream);
public:
    static Ref<AudioTrackPrivateMediaStream> create(MediaStreamTrackPrivate& streamTrack)
    {
        return adoptRef(*new AudioTrackPrivateMediaStream(streamTrack));
    }
    ~AudioTrackPrivateMediaStream();

    void setTrackIndex(int index) { m_index = index; }
    void setAudioOutputDevice(const String&);

    MediaStreamTrackPrivate& streamTrack() { return m_streamTrack.get(); }

    void clear();

    void play();
    void pause();
    bool isPlaying() const { return m_isPlaying; }
    bool shouldPlay() const { return m_shouldPlay; }

    void setVolume(float);
    float volume() const;

    void setMuted(bool);
    bool muted() const { return m_muted; }

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "AudioTrackPrivateMediaStream"_s; }
#endif

private:
    explicit AudioTrackPrivateMediaStream(MediaStreamTrackPrivate&);

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    static RefPtr<AudioMediaStreamTrackRenderer> createRenderer(AudioTrackPrivateMediaStream&);

    // AudioTrackPrivate
    Kind kind() const final { return Kind::Main; }
    std::optional<AtomString> trackUID() const { return AtomString { m_streamTrack->id() }; }
    AtomString label() const final { return AtomString { m_streamTrack->label() }; }
    int trackIndex() const final { return m_index; }
    bool isBackedByMediaStreamTrack() const final { return true; }

    // MediaStreamTrackPrivateObserver
    void trackEnded(MediaStreamTrackPrivate&) final;
    void trackMutedChanged(MediaStreamTrackPrivate&)  final;
    void trackEnabledChanged(MediaStreamTrackPrivate&)  final;
    void trackSettingsChanged(MediaStreamTrackPrivate&) final { }

    // RealtimeMediaSource::AudioSampleObserver
    void audioSamplesAvailable(const MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final;

    void startRenderer();
    void stopRenderer();
    void updateRenderer();
    void createNewRenderer();

    // Main thread writable members
    bool m_isPlaying { false };
    bool m_shouldPlay { false };
    bool m_muted { false };
    bool m_isCleared { false };

    const Ref<MediaStreamTrackPrivate> m_streamTrack;
    const Ref<RealtimeMediaSource> m_audioSource;
    int m_index { 0 };

    // Audio thread members
    RefPtr<AudioMediaStreamTrackRenderer> m_renderer;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioTrackPrivateMediaStream)
    static bool isType(const WebCore::AudioTrackPrivate& track) { return track.isBackedByMediaStreamTrack(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_STREAM)
