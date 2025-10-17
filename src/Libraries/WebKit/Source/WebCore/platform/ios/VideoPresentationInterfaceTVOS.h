/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

#if PLATFORM(APPLETV)

#include "VideoPresentationInterfaceIOS.h"

namespace WebCore {

class VideoPresentationInterfaceTVOS final : public VideoPresentationInterfaceIOS {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VideoPresentationInterfaceTVOS);
public:
    WEBCORE_EXPORT static Ref<VideoPresentationInterfaceTVOS> create(PlaybackSessionInterfaceIOS&);

    // VideoPresentationInterfaceIOS
    void hasVideoChanged(bool) final { }
    AVPlayerViewController *avPlayerViewController() const final { return { }; }
    bool mayAutomaticallyShowVideoPictureInPicture() const final { return false; }
    bool isPlayingVideoInEnhancedFullscreen() const final { return false; }
    bool pictureInPictureWasStartedWhenEnteringBackground() const final { return false; }
    
private:
    VideoPresentationInterfaceTVOS(PlaybackSessionInterfaceIOS&);

    // VideoPresentationInterfaceIOS
    void updateRouteSharingPolicy() final { }
    void setupPlayerViewController() final { }
    void invalidatePlayerViewController() final { }
    UIViewController *playerViewController() const final { return { }; }
    void presentFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) final;
    void dismissFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) final;
    void tryToStartPictureInPicture() final { }
    void stopPictureInPicture() final { }
    void setShowsPlaybackControls(bool) final { }
    void setContentDimensions(const FloatSize&) final { }
    void setAllowsPictureInPicturePlayback(bool) final { }
    bool isExternalPlaybackActive() const final { return false; }
    bool willRenderToLayer() const final { return true; }
};

} // namespace WebCore

#endif // PLATFORM(APPLETV)
