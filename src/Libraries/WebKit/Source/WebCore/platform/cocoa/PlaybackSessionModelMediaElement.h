/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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

#if PLATFORM(IOS_FAMILY) || (PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE))

#include "EventListener.h"
#include "HTMLMediaElementEnums.h"
#include "PlaybackSessionModel.h"
#if ENABLE(LINEAR_MEDIA_PLAYER)
#include "SpatialVideoMetadata.h"
#endif
#include <wtf/CheckedPtr.h>
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioTrack;
class HTMLMediaElement;
class TextTrack;

class PlaybackSessionModelMediaElement final
    : public PlaybackSessionModel
    , public EventListener
    , public CanMakeCheckedPtr<PlaybackSessionModelMediaElement> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PlaybackSessionModelMediaElement, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PlaybackSessionModelMediaElement);
public:
    static Ref<PlaybackSessionModelMediaElement> create()
    {
        return adoptRef(*new PlaybackSessionModelMediaElement());
    }
    WEBCORE_EXPORT virtual ~PlaybackSessionModelMediaElement();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    WEBCORE_EXPORT void setMediaElement(HTMLMediaElement*);
    HTMLMediaElement* mediaElement() const { return m_mediaElement.get(); }

    WEBCORE_EXPORT void mediaEngineChanged();

    WEBCORE_EXPORT void handleEvent(WebCore::ScriptExecutionContext&, WebCore::Event&) final;
    void updateForEventName(const AtomString&);
    WEBCORE_EXPORT void updateAll();

    WEBCORE_EXPORT void addClient(PlaybackSessionModelClient&);
    WEBCORE_EXPORT void removeClient(PlaybackSessionModelClient&);
    WEBCORE_EXPORT void play() final;
    WEBCORE_EXPORT void pause() final;
    WEBCORE_EXPORT void togglePlayState() final;
    WEBCORE_EXPORT void beginScrubbing() final;
    WEBCORE_EXPORT void endScrubbing() final;
    WEBCORE_EXPORT void seekToTime(double time, double toleranceBefore, double toleranceAfter) final;
    WEBCORE_EXPORT void fastSeek(double time) final;
    WEBCORE_EXPORT void beginScanningForward() final;
    WEBCORE_EXPORT void beginScanningBackward() final;
    WEBCORE_EXPORT void endScanning() final;
    WEBCORE_EXPORT void setDefaultPlaybackRate(double) final;
    WEBCORE_EXPORT void setPlaybackRate(double) final;
    WEBCORE_EXPORT void selectAudioMediaOption(uint64_t index) final;
    WEBCORE_EXPORT void selectLegibleMediaOption(uint64_t index) final;
    WEBCORE_EXPORT void togglePictureInPicture() final;
    WEBCORE_EXPORT void enterInWindowFullscreen() final;
    WEBCORE_EXPORT void exitInWindowFullscreen() final;
    WEBCORE_EXPORT void enterFullscreen() final;
    WEBCORE_EXPORT void exitFullscreen() final;
    WEBCORE_EXPORT void toggleMuted() final;
    WEBCORE_EXPORT void setMuted(bool) final;
    WEBCORE_EXPORT void setVolume(double) final;
    WEBCORE_EXPORT void setPlayingOnSecondScreen(bool) final;
#if HAVE(SPATIAL_TRACKING_LABEL)
    WEBCORE_EXPORT const String& spatialTrackingLabel() const final;
    WEBCORE_EXPORT void setSpatialTrackingLabel(const String&) final;
#endif
    WEBCORE_EXPORT void sendRemoteCommand(PlatformMediaSession::RemoteControlCommandType, const PlatformMediaSession::RemoteCommandArgument&) final;
    void setVideoReceiverEndpoint(const VideoReceiverEndpoint&) final { }

    double duration() const final;
    double currentTime() const final;
    double bufferedTime() const final;
    bool isPlaying() const final;
    bool isStalled() const final;
    bool isScrubbing() const final { return false; }
    double defaultPlaybackRate() const final;
    double playbackRate() const final;
    Ref<TimeRanges> seekableRanges() const final;
    double seekableTimeRangesLastModifiedTime() const final;
    double liveUpdateInterval() const final;
    bool canPlayFastReverse() const final;
    Vector<MediaSelectionOption> audioMediaSelectionOptions() const final;
    uint64_t audioMediaSelectedIndex() const final;
    Vector<MediaSelectionOption> legibleMediaSelectionOptions() const final;
    WEBCORE_EXPORT uint64_t legibleMediaSelectedIndex() const final;
    bool externalPlaybackEnabled() const final;
    ExternalPlaybackTargetType externalPlaybackTargetType() const final;
    String externalPlaybackLocalizedDeviceName() const final;
    bool wirelessVideoPlaybackDisabled() const final;
    bool isMuted() const final;
    double volume() const final;
    bool isPictureInPictureSupported() const final;
    bool isPictureInPictureActive() const final;
    bool isInWindowFullscreenActive() const final;
    AudioSessionSoundStageSize soundStageSize() const final { return m_soundStageSize; }
    void setSoundStageSize(AudioSessionSoundStageSize size) final { m_soundStageSize = size; }

private:
    WEBCORE_EXPORT PlaybackSessionModelMediaElement();

    void progressEventTimerFired();
    static const Vector<AtomString>& observedEventNames();
    const AtomString& eventNameAll();

#if !RELEASE_LOG_DISABLED
    uint64_t logIdentifier() const final;
    const Logger* loggerPtr() const final;
#endif

    RefPtr<HTMLMediaElement> m_mediaElement;
    bool m_isListening { false };
    UncheckedKeyHashSet<CheckedPtr<PlaybackSessionModelClient>> m_clients;
    Vector<RefPtr<TextTrack>> m_legibleTracksForMenu;
    Vector<RefPtr<AudioTrack>> m_audioTracksForMenu;
    AudioSessionSoundStageSize m_soundStageSize;
#if ENABLE(LINEAR_MEDIA_PLAYER)
    std::optional<SpatialVideoMetadata> m_spatialVideoMetadata;
#endif

    double playbackStartedTime() const;
    void updateMediaSelectionOptions();
    void updateMediaSelectionIndices();
};

}

#endif
