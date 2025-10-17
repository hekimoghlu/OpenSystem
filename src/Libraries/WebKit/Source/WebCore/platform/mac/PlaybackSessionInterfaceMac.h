/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#if PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)

#include "HTMLMediaElementEnums.h"
#include "PlaybackSessionModel.h"
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakObjCPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS WebPlaybackControlsManager;

namespace WebCore {
class IntRect;
class PlaybackSessionModel;

class WEBCORE_EXPORT PlaybackSessionInterfaceMac final
    : public PlaybackSessionModelClient
    , public RefCounted<PlaybackSessionInterfaceMac>
    , public CanMakeCheckedPtr<PlaybackSessionInterfaceMac> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PlaybackSessionInterfaceMac, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PlaybackSessionInterfaceMac);
public:
    static Ref<PlaybackSessionInterfaceMac> create(PlaybackSessionModel&);
    virtual ~PlaybackSessionInterfaceMac();
    PlaybackSessionModel* playbackSessionModel() const;

    bool isInWindowFullscreenActive() const;
    void enterInWindowFullscreen();
    void exitInWindowFullscreen();

    // PlaybackSessionModelClient
    void durationChanged(double) final;
    void currentTimeChanged(double /*currentTime*/, double /*anchorTime*/) final;
    void rateChanged(OptionSet<PlaybackSessionModel::PlaybackState>, double /* playbackRate */, double /* defaultPlaybackRate */) final;
    void seekableRangesChanged(const TimeRanges&, double /*lastModifiedTime*/, double /*liveUpdateInterval*/) final;
    void audioMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& /*options*/, uint64_t /*selectedIndex*/) final;
    void legibleMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& /*options*/, uint64_t /*selectedIndex*/) final;
    void audioMediaSelectionIndexChanged(uint64_t) final;
    void legibleMediaSelectionIndexChanged(uint64_t) final;
    void externalPlaybackChanged(bool /* enabled */, PlaybackSessionModel::ExternalPlaybackTargetType, const String& /* localizedDeviceName */) final;
    void isPictureInPictureSupportedChanged(bool) final;
    void ensureControlsManager() final;

#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    void setPlayBackControlsManager(WebPlaybackControlsManager *);
    WebPlaybackControlsManager *playBackControlsManager();

    void updatePlaybackControlsManagerCanTogglePictureInPicture();
#endif
    void willBeginScrubbing();
    void beginScrubbing();
    void endScrubbing();

    void swapFullscreenModesWith(PlaybackSessionInterfaceMac&) { }

    void invalidate();

#if !RELEASE_LOG_DISABLED
    uint64_t logIdentifier() const;
    const Logger* loggerPtr() const;
    ASCIILiteral logClassName() const { return "PlaybackSessionInterfaceMac"_s; };
    WTFLogChannel& logChannel() const;
#endif

private:
    PlaybackSessionInterfaceMac(PlaybackSessionModel&);

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final;
    uint32_t checkedPtrCountWithoutThreadCheck() const final;
    void incrementCheckedPtrCount() const final;
    void decrementCheckedPtrCount() const final;

    WeakPtr<PlaybackSessionModel> m_playbackSessionModel;
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    WeakObjCPtr<WebPlaybackControlsManager> m_playbackControlsManager;

    void updatePlaybackControlsManagerTiming(double currentTime, double anchorTime, double playbackRate, bool isPlaying);
#endif
};

}

#endif // PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)
