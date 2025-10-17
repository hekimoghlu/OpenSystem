/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

#if HAVE(AVKIT_CONTENT_SOURCE)

#include "VideoPresentationInterfaceIOS.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class VideoPresentationInterfaceAVKit final : public VideoPresentationInterfaceIOS {
    WTF_MAKE_TZONE_ALLOCATED(VideoPresentationInterfaceAVKit);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VideoPresentationInterfaceAVKit);
public:
    WEBCORE_EXPORT static Ref<VideoPresentationInterfaceAVKit> create(PlaybackSessionInterfaceIOS&);
    ~VideoPresentationInterfaceAVKit();

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const { return "VideoPresentationInterfaceAVKit"_s; };
#endif

private:
    VideoPresentationInterfaceAVKit(PlaybackSessionInterfaceIOS&);

    // VideoPresentationInterfaceIOS overrides
    bool pictureInPictureWasStartedWhenEnteringBackground() const final { return false; }
    bool mayAutomaticallyShowVideoPictureInPicture() const final { return false; }
    bool isPlayingVideoInEnhancedFullscreen() const final { return false; }
    void setupFullscreen(const FloatRect&, const FloatSize&, UIView*, HTMLMediaElementEnums::VideoFullscreenMode, bool, bool, bool) final { }
    void hasVideoChanged(bool) final { }
    void finalizeSetup() final { }
    void updateRouteSharingPolicy() final { }
    void setupPlayerViewController() final { }
    void invalidatePlayerViewController() final { }
    UIViewController *playerViewController() const final { return nullptr; }
    void tryToStartPictureInPicture() final { }
    void stopPictureInPicture() final { }
    void presentFullscreen(bool, Function<void(BOOL, NSError *)>&&) final { }
    void dismissFullscreen(bool, Function<void(BOOL, NSError *)>&&) final { }
    void setShowsPlaybackControls(bool) final { }
    void setContentDimensions(const FloatSize&) final { }
    void setAllowsPictureInPicturePlayback(bool) final { }
    bool isExternalPlaybackActive() const final { return false; }
    bool willRenderToLayer() const final { return false; }
    AVPlayerViewController *avPlayerViewController() const final { return nullptr; }
    CALayer *captionsLayer() final { return nullptr; }
    void setupCaptionsLayer(CALayer *, const FloatSize&) final { }
#if ENABLE(LINEAR_MEDIA_PLAYER)
    LMPlayableViewController *playableViewController() final { return nullptr; }
#endif
    void setSpatialImmersive(bool) final { }
};

} // namespace WebCore

#endif // HAVE(AVKIT_CONTENT_SOURCE)
