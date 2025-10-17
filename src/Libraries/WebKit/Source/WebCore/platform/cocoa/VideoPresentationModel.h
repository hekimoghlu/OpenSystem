/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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

#if ENABLE(VIDEO_PRESENTATION_MODE)

#include "AudioSession.h"
#include "FloatRect.h"
#include "HTMLMediaElementEnums.h"
#include "MediaPlayerEnums.h"
#include "MediaPlayerIdentifier.h"
#include "PlaybackSessionModel.h"
#include <wtf/CheckedPtr.h>
#include <wtf/CompletionHandler.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS AVPlayerViewController;
OBJC_CLASS UIViewController;
#endif

namespace WebCore {
class VideoPresentationModelClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::VideoPresentationModelClient> : std::true_type { };
}

namespace WTF {
class MachSendRight;
}

namespace WebCore {

class VideoPresentationModelClient;

class VideoPresentationModel
    : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<VideoPresentationModel> {
public:
    virtual ~VideoPresentationModel() = default;
    virtual void addClient(VideoPresentationModelClient&) = 0;
    virtual void removeClient(VideoPresentationModelClient&)= 0;

    virtual void requestFullscreenMode(HTMLMediaElementEnums::VideoFullscreenMode, bool finishedWithMedia = false) = 0;
    virtual void setVideoLayerFrame(FloatRect) = 0;
    virtual void setVideoLayerGravity(MediaPlayerEnums::VideoGravity) = 0;
    virtual void setVideoFullscreenFrame(FloatRect) = 0;
    virtual void fullscreenModeChanged(HTMLMediaElementEnums::VideoFullscreenMode) = 0;

    virtual FloatSize videoDimensions() const = 0;
    virtual bool hasVideo() const = 0;

    virtual void willEnterPictureInPicture() = 0;
    virtual void didEnterPictureInPicture() = 0;
    virtual void failedToEnterPictureInPicture() = 0;
    virtual void willExitPictureInPicture() = 0;
    virtual void didExitPictureInPicture() = 0;

    virtual void requestUpdateInlineRect() { };
    virtual void requestVideoContentLayer() { };
    virtual void returnVideoContentLayer() { };
    virtual void returnVideoView() { };
    virtual void didSetupFullscreen() { };
    virtual void didEnterFullscreen(const FloatSize&) { };
    virtual void failedToEnterFullscreen() { };
    virtual void willExitFullscreen() { };
    virtual void didExitFullscreen() { };
    virtual void didCleanupFullscreen() { };
    virtual void fullscreenMayReturnToInline() { };
    virtual void setRequiresTextTrackRepresentation(bool) { }
    virtual void setTextTrackRepresentationBounds(const IntRect&) { }

    virtual void requestRouteSharingPolicyAndContextUID(CompletionHandler<void(RouteSharingPolicy, String)>&& completionHandler) { completionHandler(RouteSharingPolicy::Default, emptyString()); }

#if PLATFORM(IOS_FAMILY)
    virtual UIViewController *presentingViewController() { return nullptr; }
    virtual RetainPtr<UIViewController> createVideoFullscreenViewController(AVPlayerViewController *) { return nullptr; }
#endif

#if !RELEASE_LOG_DISABLED
    virtual uint64_t logIdentifier() const { return 0; }
    virtual uint64_t nextChildIdentifier() const { return logIdentifier(); }
    virtual const Logger* loggerPtr() const { return nullptr; }
#endif
};

class VideoPresentationModelClient : public CanMakeWeakPtr<VideoPresentationModelClient> {
public:
    virtual ~VideoPresentationModelClient() = default;

    // CheckedPtr interface
    virtual uint32_t checkedPtrCount() const = 0;
    virtual uint32_t checkedPtrCountWithoutThreadCheck() const = 0;
    virtual void incrementCheckedPtrCount() const = 0;
    virtual void decrementCheckedPtrCount() const = 0;

    virtual void hasVideoChanged(bool) { }
    virtual void videoDimensionsChanged(const FloatSize&) { }
    virtual void willEnterPictureInPicture() { }
    virtual void didEnterPictureInPicture() { }
    virtual void failedToEnterPictureInPicture() { }
    virtual void willExitPictureInPicture() { }
    virtual void didExitPictureInPicture() { }
    virtual void setPlayerIdentifier(std::optional<MediaPlayerIdentifier>) { }
    virtual void documentVisibilityChanged(bool) { }
    virtual void audioSessionCategoryChanged(AudioSessionCategory, AudioSessionMode, RouteSharingPolicy) { }
};

} // namespace WebCore

#endif // ENABLE(VIDEO_PRESENTATION_MODE)
