/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#include "Connection.h"
#include "MessageReceiver.h"
#include "PlaybackSessionContextIdentifier.h"
#include <WebCore/EventListener.h>
#include <WebCore/HTMLMediaElementEnums.h>
#include <WebCore/PlatformCALayer.h>
#include <WebCore/VideoPresentationModelVideoElement.h>
#include <wtf/CheckedRef.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>

namespace IPC {
class Connection;
class Decoder;
class MessageReceiver;
}

namespace WTF {
class MachSendRight;
}

namespace WebCore {
class FloatSize;
class Node;
class ShareableBitmapHandle;
}

namespace WebKit {

class LayerHostingContext;
class WebPage;
class PlaybackSessionInterfaceContext;
class PlaybackSessionManager;
class VideoPresentationManager;

class VideoPresentationInterfaceContext final
    : public RefCounted<VideoPresentationInterfaceContext>
    , public WebCore::VideoPresentationModelClient
    , public CanMakeCheckedPtr<VideoPresentationInterfaceContext> {
    WTF_MAKE_TZONE_ALLOCATED(VideoPresentationInterfaceContext);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VideoPresentationInterfaceContext);
public:
    static Ref<VideoPresentationInterfaceContext> create(VideoPresentationManager& manager, PlaybackSessionContextIdentifier contextId)
    {
        return adoptRef(*new VideoPresentationInterfaceContext(manager, contextId));
    }
    virtual ~VideoPresentationInterfaceContext();

    LayerHostingContext* layerHostingContext() { return m_layerHostingContext.get(); }
    void setLayerHostingContext(std::unique_ptr<LayerHostingContext>&&);

    enum class AnimationType { None, IntoFullscreen, FromFullscreen };
    AnimationType animationState() const { return m_animationType; }
    void setAnimationState(AnimationType flag) { m_animationType = flag; }

    bool targetIsFullscreen() const { return m_targetIsFullscreen; }
    void setTargetIsFullscreen(bool flag) { m_targetIsFullscreen = flag; }

    WebCore::HTMLMediaElementEnums::VideoFullscreenMode fullscreenMode() const { return m_fullscreenMode; }
    void setFullscreenMode(WebCore::HTMLMediaElementEnums::VideoFullscreenMode mode) { m_fullscreenMode = mode; }

    bool fullscreenStandby() const { return m_fullscreenStandby; }
    void setFullscreenStandby(bool value) { m_fullscreenStandby = value; }

    bool isFullscreen() const { return m_isFullscreen; }
    void setIsFullscreen(bool flag) { m_isFullscreen = flag; }

    RetainPtr<CALayer> rootLayer() const { return m_rootLayer; }
    void setRootLayer(RetainPtr<CALayer>);

private:
    // VideoPresentationModelClient
    void hasVideoChanged(bool) override;
    void documentVisibilityChanged(bool) override;
    void audioSessionCategoryChanged(WebCore::AudioSessionCategory, WebCore::AudioSessionMode, WebCore::RouteSharingPolicy) final;

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    void videoDimensionsChanged(const WebCore::FloatSize&) override;
    void setPlayerIdentifier(std::optional<WebCore::MediaPlayerIdentifier>) final;

    VideoPresentationInterfaceContext(VideoPresentationManager&, PlaybackSessionContextIdentifier);

    WeakPtr<VideoPresentationManager> m_manager;
    PlaybackSessionContextIdentifier m_contextId;
    std::unique_ptr<LayerHostingContext> m_layerHostingContext;
    AnimationType m_animationType { AnimationType::None };
    bool m_targetIsFullscreen { false };
    WebCore::HTMLMediaElementEnums::VideoFullscreenMode m_fullscreenMode { WebCore::HTMLMediaElementEnums::VideoFullscreenModeNone };
    bool m_fullscreenStandby { false };
    bool m_isFullscreen { false };
    RetainPtr<CALayer> m_rootLayer;
};

class VideoPresentationManager
    : public RefCounted<VideoPresentationManager>
    , public CanMakeWeakPtr<VideoPresentationManager>
    , private IPC::MessageReceiver {
public:
    USING_CAN_MAKE_WEAKPTR(CanMakeWeakPtr<VideoPresentationManager>);

    static Ref<VideoPresentationManager> create(WebPage&, PlaybackSessionManager&);
    virtual ~VideoPresentationManager();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void invalidate();

    bool hasVideoPlayingInPictureInPicture() const;

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void setupRemoteLayerHosting(WebCore::HTMLVideoElement&);
    void willRemoveLayerForID(PlaybackSessionContextIdentifier);

    void swapFullscreenModes(WebCore::HTMLVideoElement&, WebCore::HTMLVideoElement&);

    // Interface to WebChromeClient
    bool canEnterVideoFullscreen(WebCore::HTMLMediaElementEnums::VideoFullscreenMode) const;
    bool supportsVideoFullscreen(WebCore::HTMLMediaElementEnums::VideoFullscreenMode) const;
    bool supportsVideoFullscreenStandby() const;
    void enterVideoFullscreenForVideoElement(WebCore::HTMLVideoElement&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode, bool standby);
    void exitVideoFullscreenForVideoElement(WebCore::HTMLVideoElement&, WTF::CompletionHandler<void(bool)>&& = [](bool) { });
    void exitVideoFullscreenToModeWithoutAnimation(WebCore::HTMLVideoElement&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode);
    void setVideoFullscreenMode(WebCore::HTMLVideoElement&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode);
    void clearVideoFullscreenMode(WebCore::HTMLVideoElement&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode);
    void updateTextTrackRepresentationForVideoElement(WebCore::HTMLVideoElement&, WebCore::ShareableBitmapHandle&&);
    void setTextTrackRepresentationContentScaleForVideoElement(WebCore::HTMLVideoElement&, float scale);
    void setTextTrackRepresentationIsHiddenForVideoElement(WebCore::HTMLVideoElement&, bool hidden);

    bool videoElementInPictureInPicture() const { return !!m_videoElementInPictureInPicture; }

protected:
    friend class VideoPresentationInterfaceContext;

    explicit VideoPresentationManager(WebPage&, PlaybackSessionManager&);

    typedef std::tuple<Ref<WebCore::VideoPresentationModelVideoElement>, Ref<VideoPresentationInterfaceContext>> ModelInterfaceTuple;
    ModelInterfaceTuple createModelAndInterface(PlaybackSessionContextIdentifier, bool createLayerHostingContext);
    const ModelInterfaceTuple& ensureModelAndInterface(PlaybackSessionContextIdentifier, bool createLayerHostingContext = true);
    Ref<WebCore::VideoPresentationModelVideoElement> ensureModel(PlaybackSessionContextIdentifier);
    Ref<VideoPresentationInterfaceContext> ensureInterface(PlaybackSessionContextIdentifier);
    void removeContext(PlaybackSessionContextIdentifier);
    void addClientForContext(PlaybackSessionContextIdentifier);
    void removeClientForContext(PlaybackSessionContextIdentifier);

    // Interface to VideoPresentationInterfaceContext
    void hasVideoChanged(PlaybackSessionContextIdentifier, bool hasVideo);
    void documentVisibilityChanged(PlaybackSessionContextIdentifier, bool isDocumentVisible);
    void videoDimensionsChanged(PlaybackSessionContextIdentifier, const WebCore::FloatSize&);
    void setPlayerIdentifier(PlaybackSessionContextIdentifier, std::optional<WebCore::MediaPlayerIdentifier>);
    void audioSessionCategoryChanged(PlaybackSessionContextIdentifier, WebCore::AudioSessionCategory, WebCore::AudioSessionMode, WebCore::RouteSharingPolicy);

    // Messages from VideoPresentationManagerProxy
    void requestFullscreenMode(PlaybackSessionContextIdentifier, WebCore::HTMLMediaElementEnums::VideoFullscreenMode, bool finishedWithMedia);
    void requestUpdateInlineRect(PlaybackSessionContextIdentifier);
    void requestVideoContentLayer(PlaybackSessionContextIdentifier);
    void returnVideoContentLayer(PlaybackSessionContextIdentifier);
#if !PLATFORM(IOS_FAMILY)
    void didSetupFullscreen(PlaybackSessionContextIdentifier);
#endif
    void willExitFullscreen(PlaybackSessionContextIdentifier);
    void didExitFullscreen(PlaybackSessionContextIdentifier);
    void didEnterFullscreen(PlaybackSessionContextIdentifier, std::optional<WebCore::FloatSize>);
    void failedToEnterFullscreen(PlaybackSessionContextIdentifier);
    void didCleanupFullscreen(PlaybackSessionContextIdentifier);
    void setVideoLayerFrameFenced(PlaybackSessionContextIdentifier, WebCore::FloatRect bounds, WTF::MachSendRight&&);
    void setVideoLayerGravityEnum(PlaybackSessionContextIdentifier, unsigned gravity);
    void setVideoFullscreenFrame(PlaybackSessionContextIdentifier, WebCore::FloatRect);
    void fullscreenModeChanged(PlaybackSessionContextIdentifier, WebCore::HTMLMediaElementEnums::VideoFullscreenMode);
    void fullscreenMayReturnToInline(PlaybackSessionContextIdentifier, bool isPageVisible);
    void requestRouteSharingPolicyAndContextUID(PlaybackSessionContextIdentifier, CompletionHandler<void(WebCore::RouteSharingPolicy, String)>&&);
    void ensureUpdatedVideoDimensions(PlaybackSessionContextIdentifier, WebCore::FloatSize existingVideoDimensions);

    void setCurrentlyInFullscreen(VideoPresentationInterfaceContext&, bool);
    void setRequiresTextTrackRepresentation(PlaybackSessionContextIdentifier, bool);
    void setTextTrackRepresentationBounds(PlaybackSessionContextIdentifier, const WebCore::IntRect&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const;
    uint64_t logIdentifier() const;
    ASCIILiteral logClassName() const;
    WTFLogChannel& logChannel() const;
#endif

    WeakPtr<WebPage> m_page;
    Ref<PlaybackSessionManager> m_playbackSessionManager;
    WeakHashMap<WebCore::HTMLVideoElement, PlaybackSessionContextIdentifier> m_videoElements;
    HashMap<PlaybackSessionContextIdentifier, ModelInterfaceTuple> m_contextMap;
    HashMap<PlaybackSessionContextIdentifier, int> m_clientCounts;
    WeakPtr<WebCore::HTMLVideoElement> m_videoElementInPictureInPicture;
    bool m_currentlyInFullscreen { false };
};

} // namespace WebKit

#endif // ENABLE(VIDEO_PRESENTATION_MODE)
