/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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

#if PLATFORM(IOS_FAMILY) && HAVE(AVKIT)

#include "VideoPresentationInterfaceIOS.h"
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVPlayerViewController;
OBJC_CLASS WebAVPlayerController;
OBJC_CLASS WebAVPlayerLayerView;
OBJC_CLASS WebAVPlayerLayer;
OBJC_CLASS WebAVPlayerViewController;
OBJC_CLASS WebAVPlayerViewControllerDelegate;

namespace WebCore {

class PlaybackSessionInterfaceIOS;

class VideoPresentationInterfaceAVKitLegacy final : public VideoPresentationInterfaceIOS {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(VideoPresentationInterfaceAVKitLegacy, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VideoPresentationInterfaceAVKitLegacy);
public:
    WEBCORE_EXPORT static Ref<VideoPresentationInterfaceAVKitLegacy> create(PlaybackSessionInterfaceIOS&);
    WEBCORE_EXPORT ~VideoPresentationInterfaceAVKitLegacy();

    WEBCORE_EXPORT void hasVideoChanged(bool) final;

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const { return "VideoPresentationInterfaceAVKitLegacy"_s; };
#endif

    WEBCORE_EXPORT AVPlayerViewController *avPlayerViewController() const final;
    WEBCORE_EXPORT void setupFullscreen(const FloatRect& initialRect, const FloatSize& videoDimensions, UIView* parentView, HTMLMediaElementEnums::VideoFullscreenMode, bool allowsPictureInPicturePlayback, bool standby, bool blocksReturnToFullscreenFromPictureInPicture);
    WEBCORE_EXPORT bool pictureInPictureWasStartedWhenEnteringBackground() const final;
    WEBCORE_EXPORT void setPlayerIdentifier(std::optional<MediaPlayerIdentifier>) final;
    WEBCORE_EXPORT bool mayAutomaticallyShowVideoPictureInPicture() const;
    bool isPlayingVideoInEnhancedFullscreen() const;
    bool allowsPictureInPicturePlayback() const { return m_allowsPictureInPicturePlayback; }
    void presentFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) final;
    void dismissFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) final;

    // VideoFullscreenCaptions:
    WEBCORE_EXPORT void setupCaptionsLayer(CALayer *parent, const WebCore::FloatSize&) final;

private:
    WEBCORE_EXPORT VideoPresentationInterfaceAVKitLegacy(PlaybackSessionInterfaceIOS&);

    WebAVPlayerLayer *fullscreenPlayerLayer() const;

    void updateRouteSharingPolicy() final;
    void setupPlayerViewController() final;
    void invalidatePlayerViewController() final;
    UIViewController *playerViewController() const final;
    void tryToStartPictureInPicture() final;
    void stopPictureInPicture() final;
    void setShowsPlaybackControls(bool) final;
    void setContentDimensions(const FloatSize&) final;
    void setAllowsPictureInPicturePlayback(bool) final;
    bool isExternalPlaybackActive() const final;
    bool willRenderToLayer() const final;
    void transferVideoViewToFullscreen() final;
    void returnVideoView() final;

    RetainPtr<WebAVPlayerViewControllerDelegate> m_playerViewControllerDelegate;
    RetainPtr<WebAVPlayerViewController> m_playerViewController;
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY) && HAVE(AVKIT)
