/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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

#if PLATFORM(COCOA)

#include "HTMLMediaElementEnums.h"
#include "PlaybackSessionModel.h"
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMallocInlines.h>

OBJC_CLASS AVPlayerViewController;
OBJC_CLASS UIView;

namespace WebCore {

class FloatRect;
class FloatSize;

class NullPlaybackSessionInterface final
    : public PlaybackSessionModelClient
    , public RefCounted<NullPlaybackSessionInterface>
    , public CanMakeCheckedPtr<NullPlaybackSessionInterface> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(NullPlaybackSessionInterface);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(NullPlaybackSessionInterface);
public:
    static Ref<NullPlaybackSessionInterface> create(PlaybackSessionModel& model)
    {
        return adoptRef(*new NullPlaybackSessionInterface(model));
    }

    PlaybackSessionModel* playbackSessionModel() const { return m_model.get(); }

    void enterFullscreen() { }
    bool exitFullscreen(const FloatRect&) { return false; }
    void cleanupFullscreen() { }
    void invalidate() { }
    void requestHideAndExitFullscreen() { }
    void preparedToReturnToInline(bool visible, const FloatRect& inlineRect) { }
    void preparedToExitFullscreen() { }
    void setHasVideoContentLayer(bool) { }
    void setInlineRect(const FloatRect&, bool) { }
    void preparedToReturnToStandby() { }
    bool mayAutomaticallyShowVideoPictureInPicture() const { return false; }
    void applicationDidBecomeActive() { }
    void setMode(HTMLMediaElementEnums::VideoFullscreenMode, bool) { }
    bool pictureInPictureWasStartedWhenEnteringBackground() const { return false; }
    AVPlayerViewController *avPlayerViewController() const { return nullptr; }
    void swapFullscreenModesWith(NullPlaybackSessionInterface&) { }

private:
    NullPlaybackSessionInterface(PlaybackSessionModel& model)
        : m_model(model)
    {
    }

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    void durationChanged(double) final { }
    void currentTimeChanged(double, double) final { }
    void bufferedTimeChanged(double) final { }
    void rateChanged(OptionSet<PlaybackSessionModel::PlaybackState>, double, double) final { }
    void seekableRangesChanged(const TimeRanges&, double, double) final { }
    void canPlayFastReverseChanged(bool) final { }
    void audioMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>&, uint64_t) final { }
    void legibleMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>&, uint64_t) final { }
    void externalPlaybackChanged(bool, PlaybackSessionModel::ExternalPlaybackTargetType, const String&) final { }
    void wirelessVideoPlaybackDisabledChanged(bool) final { }
    void mutedChanged(bool) final { }
    void volumeChanged(double) final { }
    void modelDestroyed() final { }

    WeakPtr<PlaybackSessionModel> m_model;
};

}

#endif
