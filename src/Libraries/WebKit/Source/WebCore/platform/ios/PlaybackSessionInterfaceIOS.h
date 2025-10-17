/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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

#if PLATFORM(COCOA) && HAVE(AVKIT)

#include "EventListener.h"
#include "HTMLMediaElementEnums.h"
#include "MediaPlayerIdentifier.h"
#include "PlaybackSessionModel.h"
#include "Timer.h"
#include <functional>
#include <objc/objc.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS WKSLinearMediaPlayer;
OBJC_CLASS WebAVPlayerController;

namespace WebCore {

class IntRect;
class PlaybackSessionModel;
class WebPlaybackSessionChangeObserver;

class WEBCORE_EXPORT PlaybackSessionInterfaceIOS
    : public PlaybackSessionModelClient
    , public RefCounted<PlaybackSessionInterfaceIOS>
    , public CanMakeCheckedPtr<PlaybackSessionInterfaceIOS> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PlaybackSessionInterfaceIOS, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PlaybackSessionInterfaceIOS);
public:
    void initialize();
    virtual void invalidate();
    virtual ~PlaybackSessionInterfaceIOS();

    virtual WebAVPlayerController *playerController() const = 0;
    virtual WKSLinearMediaPlayer *linearMediaPlayer() const = 0;
    PlaybackSessionModel* playbackSessionModel() const;
    void durationChanged(double) override = 0;
    void currentTimeChanged(double currentTime, double anchorTime) override = 0;
    void bufferedTimeChanged(double) override = 0;
    void rateChanged(OptionSet<PlaybackSessionModel::PlaybackState>, double playbackRate, double defaultPlaybackRate) override = 0;
    void seekableRangesChanged(const TimeRanges&, double lastModifiedTime, double liveUpdateInterval) override = 0;
    void canPlayFastReverseChanged(bool) override = 0;
    void audioMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& options, uint64_t selectedIndex) override = 0;
    void legibleMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& options, uint64_t selectedIndex) override = 0;
    void externalPlaybackChanged(bool enabled, PlaybackSessionModel::ExternalPlaybackTargetType, const String& localizedDeviceName) override = 0;
    void wirelessVideoPlaybackDisabledChanged(bool) override = 0;
    void mutedChanged(bool) override = 0;
    void volumeChanged(double) override = 0;
    void modelDestroyed() override;

    std::optional<MediaPlayerIdentifier> playerIdentifier() const;
    void setPlayerIdentifier(std::optional<MediaPlayerIdentifier>);

    virtual void startObservingNowPlayingMetadata();
    virtual void stopObservingNowPlayingMetadata();

    virtual void swapFullscreenModesWith(PlaybackSessionInterfaceIOS&) { }

#if !RELEASE_LOG_DISABLED
    uint64_t logIdentifier() const;
    const Logger* loggerPtr() const;
    virtual ASCIILiteral logClassName() const = 0;
    WTFLogChannel& logChannel() const;
#endif

protected:
#if HAVE(SPATIAL_TRACKING_LABEL)
    void updateSpatialTrackingLabel();
#endif

    PlaybackSessionInterfaceIOS(PlaybackSessionModel&);
    PlaybackSessionModel* m_playbackSessionModel { nullptr };

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final;
    uint32_t checkedPtrCountWithoutThreadCheck() const final;
    void incrementCheckedPtrCount() const final;
    void decrementCheckedPtrCount() const final;

private:
    std::optional<MediaPlayerIdentifier> m_playerIdentifier;
#if HAVE(SPATIAL_TRACKING_LABEL)
    String m_spatialTrackingLabel;
    String m_defaultSpatialTrackingLabel;
#endif
};

} // namespace WebCore

#endif // PLATFORM(COCOA) && HAVE(AVKIT)
