/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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

#include <WebCore/EventListener.h>
#include <WebCore/HTMLMediaElement.h>
#include <WebCore/HTMLMediaElementEnums.h>
#include <WebCore/IntRect.h>
#include <WebCore/LengthBox.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebCore {
class IntRect;
class Element;
class WeakPtrImplWithEventTargetData;
class GraphicsLayer;
class HTMLVideoElement;
class RenderImage;
}

namespace WebKit {

class WebPage;
struct FullScreenMediaDetails;

class WebFullScreenManager final : public WebCore::EventListener {
public:
    static Ref<WebFullScreenManager> create(WebPage&);
    virtual ~WebFullScreenManager();

    void invalidate();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    bool supportsFullScreenForElement(const WebCore::Element&, bool withKeyboard);
    void enterFullScreenForElement(WebCore::Element&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode);
#if ENABLE(QUICKLOOK_FULLSCREEN)
    void updateImageSource(WebCore::Element&);
#endif // ENABLE(QUICKLOOK_FULLSCREEN)
    void exitFullScreenForElement(WebCore::Element*);

    void willEnterFullScreen(WebCore::HTMLMediaElementEnums::VideoFullscreenMode = WebCore::HTMLMediaElementEnums::VideoFullscreenModeStandard);
    void didEnterFullScreen();
    void willExitFullScreen();
    void didExitFullScreen();

    void saveScrollPosition();
    void restoreScrollPosition();

    WebCore::Element* element();

    void videoControlsManagerDidChange();

    bool operator==(const WebCore::EventListener& listener) const final { return this == &listener; }

protected:
    WebFullScreenManager(WebPage&);

    void setPIPStandbyElement(WebCore::HTMLVideoElement*);

    void setAnimatingFullScreen(bool);
    void requestRestoreFullScreen(CompletionHandler<void(bool)>&&);
    void requestExitFullScreen();
    void setFullscreenInsets(const WebCore::FloatBoxExtent&);
    void setFullscreenAutoHideDuration(Seconds);

    WebCore::IntRect m_initialFrame;
    WebCore::IntRect m_finalFrame;
    WebCore::IntPoint m_scrollPosition;
    float m_topContentInset { 0 };
    Ref<WebPage> m_page;
    RefPtr<WebCore::Element> m_element;
    WeakPtr<WebCore::Element, WebCore::WeakPtrImplWithEventTargetData> m_elementToRestore;
#if ENABLE(QUICKLOOK_FULLSCREEN)
    WebCore::FloatSize m_oldSize;
    double m_scaleFactor { 1 };
    double m_minEffectiveWidth { 0 };
#endif
#if ENABLE(VIDEO)
    RefPtr<WebCore::HTMLVideoElement> m_pipStandbyElement;
#endif

private:
    void close();

    void handleEvent(WebCore::ScriptExecutionContext&, WebCore::Event&) final;

    void setElement(WebCore::Element&);
    void clearElement();

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "WebFullScreenManager"_s; }
    WTFLogChannel& logChannel() const;
#endif

#if ENABLE(VIDEO)
#if ENABLE(IMAGE_ANALYSIS)
    void scheduleTextRecognitionForMainVideo();
    void endTextRecognitionForMainVideoIfNeeded();
    void mainVideoElementTextRecognitionTimerFired();
#endif
    void updateMainVideoElement();
    void setMainVideoElement(RefPtr<WebCore::HTMLVideoElement>&&);

    WeakPtr<WebCore::HTMLVideoElement> m_mainVideoElement;
#if ENABLE(IMAGE_ANALYSIS)
    RunLoop::Timer m_mainVideoElementTextRecognitionTimer;
    bool m_isPerformingTextRecognitionInMainVideo { false };
#endif
#endif // ENABLE(VIDEO)

#if ENABLE(QUICKLOOK_FULLSCREEN)
    enum class IsUpdating : bool { No, Yes };
    FullScreenMediaDetails getImageMediaDetails(CheckedPtr<WebCore::RenderImage>, IsUpdating);
    bool m_willUseQuickLookForFullscreen { false };
#endif

    bool m_closing { false };
    bool m_inWindowFullScreenMode { false };
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebKit

#endif // ENABLE(FULLSCREEN_API)
