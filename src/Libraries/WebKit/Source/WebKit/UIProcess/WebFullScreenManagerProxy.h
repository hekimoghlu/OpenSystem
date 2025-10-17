/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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

#if ENABLE(FULLSCREEN_API)

#include "FullScreenMediaDetails.h"
#include "MessageReceiver.h"
#include <WebCore/HTMLMediaElement.h>
#include <WebCore/HTMLMediaElementEnums.h>
#include <WebCore/ProcessIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/CompletionHandler.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {
class FloatSize;
class IntRect;

enum class ScreenOrientationType : uint8_t;

template <typename> class RectEdges;
using FloatBoxExtent = RectEdges<float>;
}

namespace WebKit {

class RemotePageFullscreenManagerProxy;
class WebFullScreenManagerProxy;
class WebPageProxy;
class WebProcessProxy;
struct SharedPreferencesForWebProcess;

class WebFullScreenManagerProxyClient : public CanMakeCheckedPtr<WebFullScreenManagerProxyClient> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebFullScreenManagerProxyClient);
public:
    virtual ~WebFullScreenManagerProxyClient() { }

    virtual void closeFullScreenManager() = 0;
    virtual bool isFullScreen() = 0;
#if PLATFORM(IOS_FAMILY)
    virtual void enterFullScreen(WebCore::FloatSize mediaDimensions, CompletionHandler<void(bool)>&&) = 0;
#else
    virtual void enterFullScreen(CompletionHandler<void(bool)>&&) = 0;
#endif
#if ENABLE(QUICKLOOK_FULLSCREEN)
    virtual void updateImageSource() = 0;
#endif
    virtual void exitFullScreen() = 0;
    virtual void beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) = 0;
    virtual void beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) = 0;

    virtual bool lockFullscreenOrientation(WebCore::ScreenOrientationType) { return false; }
    virtual void unlockFullscreenOrientation() { }
};

class WebFullScreenManagerProxy : public IPC::MessageReceiver, public CanMakeCheckedPtr<WebFullScreenManagerProxy>, public RefCounted<WebFullScreenManagerProxy> {
    WTF_MAKE_TZONE_ALLOCATED(WebFullScreenManagerProxy);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebFullScreenManagerProxy);
public:
    static Ref<WebFullScreenManagerProxy> create(WebPageProxy&, WebFullScreenManagerProxyClient&);
    virtual ~WebFullScreenManagerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess(const IPC::Connection&) const;

    bool isFullScreen();
    bool blocksReturnToFullscreenFromPictureInPicture() const;
#if ENABLE(VIDEO_USES_ELEMENT_FULLSCREEN)
    bool isVideoElement() const { return m_isVideoElement; }
#endif
#if ENABLE(QUICKLOOK_FULLSCREEN)
    bool isImageElement() const { return m_imageBuffer; }
    void prepareQuickLookImageURL(CompletionHandler<void(URL&&)>&&) const;
#endif // QUICKLOOK_FULLSCREEN
    void close();
    void detachFromClient();
    void attachToNewClient(WebFullScreenManagerProxyClient&);

    enum class FullscreenState : uint8_t {
        NotInFullscreen,
        EnteringFullscreen,
        InFullscreen,
        ExitingFullscreen,
    };
    FullscreenState fullscreenState() const { return m_fullscreenState; }
    void willEnterFullScreen(CompletionHandler<void(bool)>&&);
    void didEnterFullScreen();
    void willExitFullScreen();
    void didExitFullScreen();
    void setAnimatingFullScreen(bool);
    void requestRestoreFullScreen(CompletionHandler<void(bool)>&&);
    void requestExitFullScreen();
    void saveScrollPosition();
    void restoreScrollPosition();
    void setFullscreenInsets(const WebCore::FloatBoxExtent&);
    void setFullscreenAutoHideDuration(Seconds);
    void closeWithCallback(CompletionHandler<void()>&&);
    bool lockFullscreenOrientation(WebCore::ScreenOrientationType);
    void unlockFullscreenOrientation();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

private:
    WebFullScreenManagerProxy(WebPageProxy&, WebFullScreenManagerProxyClient&);

    void enterFullScreen(IPC::Connection&, bool blocksReturnToFullscreenFromPictureInPicture, FullScreenMediaDetails&&, CompletionHandler<void(bool)>&&);
#if ENABLE(QUICKLOOK_FULLSCREEN)
    void updateImageSource(FullScreenMediaDetails&&);
#endif
    void exitFullScreen();
    void beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame);
    void beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame);
    void callCloseCompletionHandlers();
    template<typename M> void sendToWebProcess(M&&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "WebFullScreenManagerProxy"_s; }
    WTFLogChannel& logChannel() const;
#endif

    WeakPtr<WebPageProxy> m_page;
    CheckedPtr<WebFullScreenManagerProxyClient> m_client;
    FullscreenState m_fullscreenState { FullscreenState::NotInFullscreen };
    bool m_blocksReturnToFullscreenFromPictureInPicture { false };
#if ENABLE(VIDEO_USES_ELEMENT_FULLSCREEN)
    bool m_isVideoElement { false };
#endif
#if ENABLE(QUICKLOOK_FULLSCREEN)
    String m_imageMIMEType;
    RefPtr<WebCore::SharedBuffer> m_imageBuffer;
#endif // QUICKLOOK_FULLSCREEN
    Vector<CompletionHandler<void()>> m_closeCompletionHandlers;
    WeakPtr<WebProcessProxy> m_fullScreenProcess;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebKit

#endif // ENABLE(FULLSCREEN_API)
