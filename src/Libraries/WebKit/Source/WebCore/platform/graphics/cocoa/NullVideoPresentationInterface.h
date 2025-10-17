/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#include "NullPlaybackSessionInterface.h"
#include "VideoFullscreenCaptions.h"
#include "VideoPresentationLayerProvider.h"
#include "VideoPresentationModel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class NullVideoPresentationInterface final
    : public VideoPresentationModelClient
    , public PlaybackSessionModelClient
    , public VideoFullscreenCaptions
    , public VideoPresentationLayerProvider
    , public RefCounted<NullVideoPresentationInterface>
    , public CanMakeCheckedPtr<NullVideoPresentationInterface> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(NullVideoPresentationInterface);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(NullVideoPresentationInterface);
public:
    static Ref<NullVideoPresentationInterface> create(NullPlaybackSessionInterface& playbackSessionInterface)
    {
        return adoptRef(*new NullVideoPresentationInterface(playbackSessionInterface));
    }

    ~NullVideoPresentationInterface() = default;
    NullPlaybackSessionInterface& playbackSessionInterface() const { return m_playbackSessionInterface.get(); }
    PlaybackSessionModel* playbackSessionModel() const { return m_playbackSessionInterface->playbackSessionModel(); }

    void setSpatialVideoMetadata(const std::optional<SpatialVideoMetadata>&) { }
    void setVideoPresentationModel(VideoPresentationModel* model) { m_videoPresentationModel = model; }
    void setupFullscreen(const FloatRect&, const FloatSize&, UIView*, HTMLMediaElementEnums::VideoFullscreenMode, bool, bool, bool) { }
    void enterFullscreen() { }
    bool exitFullscreen(const FloatRect& finalRect) { return false; }
    void exitFullscreenWithoutAnimationToMode(HTMLMediaElementEnums::VideoFullscreenMode) { }
    void cleanupFullscreen() { }
    void invalidate() { }
    void requestHideAndExitFullscreen() { }
    void preparedToReturnToInline(bool visible, const FloatRect& inlineRect) { }
    void preparedToExitFullscreen() { }
    void setHasVideoContentLayer(bool) { }
    void setInlineRect(const FloatRect&, bool visible) { }
    void preparedToReturnToStandby() { }
    bool mayAutomaticallyShowVideoPictureInPicture() const { return false; }
    void applicationDidBecomeActive() { }
    void setMode(HTMLMediaElementEnums::VideoFullscreenMode, bool) { }
    HTMLMediaElementEnums::VideoFullscreenMode mode() const { return HTMLMediaElementEnums::VideoFullscreenModeNone; }
    bool hasMode(HTMLMediaElementEnums::VideoFullscreenMode) const { return false; }
    bool pictureInPictureWasStartedWhenEnteringBackground() const { return false; }
    AVPlayerViewController *avPlayerViewController() const { return nullptr; }
    bool isPlayingVideoInEnhancedFullscreen() const { return false; }
    std::optional<MediaPlayerIdentifier> playerIdentifier() const { return std::nullopt; }
    bool changingStandbyOnly() { return false; }
    bool returningToStandby() const { return false; }

    // VideoPresentationModelClient
    void hasVideoChanged(bool) final { }
    void videoDimensionsChanged(const FloatSize&) final { }
    void setPlayerIdentifier(std::optional<MediaPlayerIdentifier>) final { }

    // PlaybackSessionModelClient
    void externalPlaybackChanged(bool, PlaybackSessionModel::ExternalPlaybackTargetType, const String&) final { }

    void swapFullscreenModesWith(NullVideoPresentationInterface&) { }

private:
    NullVideoPresentationInterface(NullPlaybackSessionInterface& playbackSessionInterface)
        : m_playbackSessionInterface(playbackSessionInterface)
    {
    }

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    Ref<NullPlaybackSessionInterface> m_playbackSessionInterface;
    ThreadSafeWeakPtr<VideoPresentationModel> m_videoPresentationModel;
};

} // namespace WebCore

#endif // PLATFORM(COCOA)
