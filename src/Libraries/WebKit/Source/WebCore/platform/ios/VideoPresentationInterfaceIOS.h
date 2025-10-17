/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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

#if PLATFORM(IOS_FAMILY)

#include "EventListener.h"
#include "HTMLMediaElementEnums.h"
#include "MediaPlayerIdentifier.h"
#include "PlatformImage.h"
#include "PlatformLayer.h"
#include "PlaybackSessionInterfaceIOS.h"
#include "SpatialVideoMetadata.h"
#include "VideoFullscreenCaptions.h"
#include "VideoPresentationLayerProvider.h"
#include "VideoPresentationModel.h"
#include <objc/objc.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/OptionSet.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

OBJC_CLASS AVPlayerViewController;
OBJC_CLASS LMPlayableViewController;
OBJC_CLASS UIImage;
OBJC_CLASS UIViewController;
OBJC_CLASS UIWindow;
OBJC_CLASS UIView;
OBJC_CLASS CALayer;
OBJC_CLASS NSError;
OBJC_CLASS WebAVPlayerController;

namespace WebCore {

class FloatRect;
class FloatSize;

class VideoPresentationInterfaceIOS
    : public VideoPresentationModelClient
    , public PlaybackSessionModelClient
    , public VideoFullscreenCaptions
    , public VideoPresentationLayerProvider
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<VideoPresentationInterfaceIOS, WTF::DestructionThread::MainRunLoop>
    , public CanMakeCheckedPtr<VideoPresentationInterfaceIOS> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(VideoPresentationInterfaceIOS, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VideoPresentationInterfaceIOS);
public:
    WEBCORE_EXPORT ~VideoPresentationInterfaceIOS();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    // VideoPresentationModelClient
    void hasVideoChanged(bool) override { }
    WEBCORE_EXPORT void videoDimensionsChanged(const FloatSize&) override;
    WEBCORE_EXPORT void setPlayerIdentifier(std::optional<MediaPlayerIdentifier>) override;
    WEBCORE_EXPORT void audioSessionCategoryChanged(WebCore::AudioSessionCategory, WebCore::AudioSessionMode, WebCore::RouteSharingPolicy) override;

    // PlaybackSessionModelClient
    WEBCORE_EXPORT void externalPlaybackChanged(bool enabled, PlaybackSessionModel::ExternalPlaybackTargetType, const String& localizedDeviceName) override;

    WEBCORE_EXPORT void setVideoPresentationModel(VideoPresentationModel*);
    PlaybackSessionInterfaceIOS& playbackSessionInterface() const { return m_playbackSessionInterface.get(); }
    PlaybackSessionModel* playbackSessionModel() const { return m_playbackSessionInterface->playbackSessionModel(); }
    virtual void setSpatialImmersive(bool) { }
    WEBCORE_EXPORT virtual void setupFullscreen(const FloatRect& initialRect, const FloatSize& videoDimensions, UIView* parentView, HTMLMediaElementEnums::VideoFullscreenMode, bool allowsPictureInPicturePlayback, bool standby, bool blocksReturnToFullscreenFromPictureInPicture);
    WEBCORE_EXPORT virtual AVPlayerViewController *avPlayerViewController() const = 0;
    WebAVPlayerController *playerController() const;
    WEBCORE_EXPORT void enterFullscreen();
    WEBCORE_EXPORT virtual bool exitFullscreen(const FloatRect& finalRect);
    WEBCORE_EXPORT void exitFullscreenWithoutAnimationToMode(HTMLMediaElementEnums::VideoFullscreenMode);
    WEBCORE_EXPORT virtual void cleanupFullscreen();
    WEBCORE_EXPORT void invalidate();
    WEBCORE_EXPORT virtual void requestHideAndExitFullscreen();
    WEBCORE_EXPORT virtual void preparedToReturnToInline(bool visible, const FloatRect& inlineRect);
    WEBCORE_EXPORT void preparedToExitFullscreen();
    WEBCORE_EXPORT void setHasVideoContentLayer(bool);
    WEBCORE_EXPORT virtual void setInlineRect(const FloatRect&, bool visible);
    WEBCORE_EXPORT void preparedToReturnToStandby();
    bool changingStandbyOnly() { return m_changingStandbyOnly; }
    WEBCORE_EXPORT void failedToRestoreFullscreen();

    enum class ExitFullScreenReason {
        DoneButtonTapped,
        FullScreenButtonTapped,
        PinchGestureHandled,
        RemoteControlStopEventReceived,
        PictureInPictureStarted
    };

    class Mode {
        HTMLMediaElementEnums::VideoFullscreenMode m_mode { HTMLMediaElementEnums::VideoFullscreenModeNone };

    public:
        Mode() = default;
        Mode(const Mode&) = default;
        Mode(HTMLMediaElementEnums::VideoFullscreenMode mode) : m_mode(mode) { }
        void operator=(HTMLMediaElementEnums::VideoFullscreenMode mode) { m_mode = mode; }
        HTMLMediaElementEnums::VideoFullscreenMode mode() const { return m_mode; }

        void setModeValue(HTMLMediaElementEnums::VideoFullscreenMode mode, bool value) { value ? setMode(mode) : clearMode(mode); }
        void setMode(HTMLMediaElementEnums::VideoFullscreenMode mode) { m_mode |= mode; }
        void clearMode(HTMLMediaElementEnums::VideoFullscreenMode mode) { m_mode &= ~mode; }
        bool hasMode(HTMLMediaElementEnums::VideoFullscreenMode mode) const { return m_mode & mode; }

        bool isPictureInPicture() const { return m_mode == HTMLMediaElementEnums::VideoFullscreenModePictureInPicture; }
        bool isFullscreen() const { return m_mode == HTMLMediaElementEnums::VideoFullscreenModeStandard; }

        void setPictureInPicture(bool value) { setModeValue(HTMLMediaElementEnums::VideoFullscreenModePictureInPicture, value); }
        void setFullscreen(bool value) { setModeValue(HTMLMediaElementEnums::VideoFullscreenModeStandard, value); }

        bool hasFullscreen() const { return hasMode(HTMLMediaElementEnums::VideoFullscreenModeStandard); }
        bool hasPictureInPicture() const { return hasMode(HTMLMediaElementEnums::VideoFullscreenModePictureInPicture); }

        bool hasVideo() const { return m_mode & (HTMLMediaElementEnums::VideoFullscreenModeStandard | HTMLMediaElementEnums::VideoFullscreenModePictureInPicture); }
    };

    RefPtr<VideoPresentationModel> videoPresentationModel() const { return m_videoPresentationModel.get(); }
    WEBCORE_EXPORT virtual bool shouldExitFullscreenWithReason(ExitFullScreenReason);
    HTMLMediaElementEnums::VideoFullscreenMode mode() const { return m_currentMode.mode(); }
    WEBCORE_EXPORT virtual bool mayAutomaticallyShowVideoPictureInPicture() const = 0;
    void prepareForPictureInPictureStop(Function<void(bool)>&& callback);
    WEBCORE_EXPORT void applicationDidBecomeActive();
    bool inPictureInPicture() const { return m_enteringPictureInPicture || m_currentMode.hasPictureInPicture(); }
    bool returningToStandby() const { return m_returningToStandby; }

    WEBCORE_EXPORT virtual void willStartPictureInPicture();
    WEBCORE_EXPORT virtual void didStartPictureInPicture();
    WEBCORE_EXPORT virtual void failedToStartPictureInPicture();
    void willStopPictureInPicture();
    WEBCORE_EXPORT virtual void didStopPictureInPicture();
    WEBCORE_EXPORT virtual void prepareForPictureInPictureStopWithCompletionHandler(void (^)(BOOL));
    virtual bool isPlayingVideoInEnhancedFullscreen() const = 0;

    WEBCORE_EXPORT void setMode(HTMLMediaElementEnums::VideoFullscreenMode, bool shouldNotifyModel);
    void clearMode(HTMLMediaElementEnums::VideoFullscreenMode, bool shouldNotifyModel);
    bool hasMode(HTMLMediaElementEnums::VideoFullscreenMode mode) const { return m_currentMode.hasMode(mode); }
    WEBCORE_EXPORT UIViewController *presentingViewController();
    UIViewController *fullscreenViewController() const { return m_viewController.get(); }
    WEBCORE_EXPORT virtual bool pictureInPictureWasStartedWhenEnteringBackground() const = 0;

    WEBCORE_EXPORT std::optional<MediaPlayerIdentifier> playerIdentifier() const;

#if ENABLE(LINEAR_MEDIA_PLAYER)
    virtual LMPlayableViewController *playableViewController() { return nil; }
#endif

    virtual void swapFullscreenModesWith(VideoPresentationInterfaceIOS&) { }

#if !RELEASE_LOG_DISABLED
    WEBCORE_EXPORT uint64_t logIdentifier() const;
    WEBCORE_EXPORT const Logger* loggerPtr() const;
    ASCIILiteral logClassName() const { return "VideoPresentationInterfaceIOS"_s; };
    WEBCORE_EXPORT WTFLogChannel& logChannel() const;
#endif

protected:
    WEBCORE_EXPORT VideoPresentationInterfaceIOS(PlaybackSessionInterfaceIOS&);

    RunLoop::Timer m_watchdogTimer;
    RetainPtr<UIView> m_parentView;
    Mode m_targetMode;
    RouteSharingPolicy m_routeSharingPolicy { RouteSharingPolicy::Default };
    String m_routingContextUID;
    ThreadSafeWeakPtr<VideoPresentationModel> m_videoPresentationModel;
    bool m_blocksReturnToFullscreenFromPictureInPicture { false };
    bool m_targetStandby { false };
    bool m_cleanupNeedsReturnVideoContentLayer { false };
    bool m_standby { false };
    Mode m_currentMode;
    bool m_enteringPictureInPicture { false };
    RetainPtr<UIWindow> m_window;
    RetainPtr<UIViewController> m_viewController;
    bool m_hasVideoContentLayer { false };
    Function<void(bool)> m_prepareToInlineCallback;
    bool m_exitingPictureInPicture { false };
    bool m_shouldReturnToFullscreenWhenStoppingPictureInPicture { false };
    bool m_enterFullscreenNeedsExitPictureInPicture { false };
    bool m_enterFullscreenNeedsEnterPictureInPicture { false };
    bool m_hasUpdatedInlineRect { false };
    bool m_inlineIsVisible { false };
    bool m_returningToStandby { false };
    bool m_exitFullscreenNeedsExitPictureInPicture { false };
    bool m_setupNeedsInlineRect { false };
    bool m_exitFullscreenNeedInlineRect { false };
    bool m_exitFullscreenNeedsReturnContentLayer { false };
    FloatRect m_inlineRect;
    bool m_shouldIgnoreAVKitCallbackAboutExitFullscreenReason { false };
    bool m_changingStandbyOnly { false };
    bool m_allowsPictureInPicturePlayback { false };
    RetainPtr<UIWindow> m_parentWindow;

    virtual void finalizeSetup();
    virtual void updateRouteSharingPolicy() = 0;
    virtual void setupPlayerViewController() = 0;
    virtual void invalidatePlayerViewController() = 0;
    virtual UIViewController *playerViewController() const = 0;
    WEBCORE_EXPORT void doSetup();

    enum class NextAction : uint8_t {
        NeedsEnterFullScreen = 1 << 0,
        NeedsExitFullScreen = 1 << 1,
    };
    using NextActions = OptionSet<NextAction>;

    WEBCORE_EXPORT void enterFullscreenHandler(BOOL success, NSError *, NextActions = NextActions());
    WEBCORE_EXPORT void exitFullscreenHandler(BOOL success, NSError *, NextActions = NextActions());
    void doEnterFullscreen();
    void doExitFullscreen();
    virtual void presentFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) = 0;
    virtual void dismissFullscreen(bool animated, Function<void(BOOL, NSError *)>&&) = 0;
    virtual void tryToStartPictureInPicture() = 0;
    virtual void stopPictureInPicture() = 0;
    virtual void setShowsPlaybackControls(bool) = 0;
    virtual void setContentDimensions(const FloatSize&) = 0;
    virtual void setAllowsPictureInPicturePlayback(bool) = 0;
    virtual bool isExternalPlaybackActive() const = 0;
    virtual bool willRenderToLayer() const = 0;
    virtual void transferVideoViewToFullscreen() { }
    WEBCORE_EXPORT virtual void returnVideoView();

#if PLATFORM(WATCHOS)
    bool m_waitingForPreparedToExit { false };
#endif
private:
    void returnToStandby();
    void watchdogTimerFired();
    void ensurePipPlacardIsShowing();

    bool m_finalizeSetupNeedsVideoContentLayer { false };
    bool m_finalizeSetupNeedsReturnVideoContentLayer { false };
    Ref<PlaybackSessionInterfaceIOS> m_playbackSessionInterface;
    RetainPtr<UIView> m_pipPlacard;
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
