/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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

#if ENABLE(LINEAR_MEDIA_PLAYER)

#include <WebCore/VideoPresentationInterfaceIOS.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS LMPlayableViewController;
OBJC_CLASS WKCaptionLayerLayoutManager;
OBJC_CLASS WKSLinearMediaPlayer;

namespace WebCore {
class PlaybackSessionInterfaceIOS;
}

namespace WebKit {

class VideoPresentationInterfaceLMK final : public WebCore::VideoPresentationInterfaceIOS {
    WTF_MAKE_TZONE_ALLOCATED(VideoPresentationInterfaceLMK);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VideoPresentationInterfaceLMK);
public:
    static Ref<VideoPresentationInterfaceLMK> create(WebCore::PlaybackSessionInterfaceIOS&);
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const { return "VideoPresentationInterfaceLMK"_s; };
#endif
    ~VideoPresentationInterfaceLMK();

    void captionsLayerBoundsChanged(const WebCore::FloatRect&);

private:
    VideoPresentationInterfaceLMK(WebCore::PlaybackSessionInterfaceIOS&);

    bool pictureInPictureWasStartedWhenEnteringBackground() const final { return false; }
    bool mayAutomaticallyShowVideoPictureInPicture() const final { return false; }
    bool isPlayingVideoInEnhancedFullscreen() const final { return false; }
    void setupFullscreen(const WebCore::FloatRect&, const WebCore::FloatSize&, UIView*, WebCore::HTMLMediaElementEnums::VideoFullscreenMode, bool, bool, bool) final;
    void hasVideoChanged(bool) final { }
    void finalizeSetup() final;
    void updateRouteSharingPolicy() final { }
    void setupPlayerViewController() final;
    void invalidatePlayerViewController() final;
    UIViewController *playerViewController() const final;
    void tryToStartPictureInPicture() final { }
    void stopPictureInPicture() final { }
    void presentFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) final;
    void dismissFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) final;
    void setShowsPlaybackControls(bool) final;
    void setContentDimensions(const WebCore::FloatSize&) final;
    void setAllowsPictureInPicturePlayback(bool) final { }
    bool isExternalPlaybackActive() const final { return false; }
    bool willRenderToLayer() const final { return false; }
    AVPlayerViewController *avPlayerViewController() const final { return nullptr; }
    CALayer *captionsLayer() final;
    void setupCaptionsLayer(CALayer *parent, const WebCore::FloatSize&) final;
    LMPlayableViewController *playableViewController() final;
    void setSpatialImmersive(bool) final;
    void swapFullscreenModesWith(VideoPresentationInterfaceIOS&) final;

    WKSLinearMediaPlayer *linearMediaPlayer() const;
    void ensurePlayableViewController();

    RetainPtr<LMPlayableViewController> m_playerViewController;

#if HAVE(SPATIAL_TRACKING_LABEL)
    String m_spatialTrackingLabel;
    RetainPtr<CALayer> m_spatialTrackingLayer;
#endif
};

} // namespace WebKit

#endif // ENABLE(LINEAR_MEDIA_PLAYER)
