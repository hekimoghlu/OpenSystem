/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#include "PageClient.h"
#include <WebCore/PlatformTextAlternatives.h>
#include <wtf/Forward.h>
#include <wtf/WeakObjCPtr.h>

@class PlatformTextAlternatives;
@class WKWebView;

namespace API {
class Attachment;
}

namespace WebCore {
class AlternativeTextUIController;
class Color;

struct AppHighlight;
}

namespace WebKit {

struct TextAnimationData;
enum class TextAnimationType : uint8_t;

class PageClientImplCocoa : public PageClient {
public:
    PageClientImplCocoa(WKWebView *);
    virtual ~PageClientImplCocoa();

    void pageClosed() override;

    void topContentInsetDidChange() final;

#if ENABLE(GPU_PROCESS)
    void gpuProcessDidFinishLaunching() override;
    void gpuProcessDidExit() override;
#endif

#if ENABLE(MODEL_PROCESS)
    void modelProcessDidFinishLaunching() override;
    void modelProcessDidExit() override;
#endif

    void themeColorWillChange() final;
    void themeColorDidChange() final;
    void underPageBackgroundColorWillChange() final;
    void underPageBackgroundColorDidChange() final;
    void sampledPageTopColorWillChange() final;
    void sampledPageTopColorDidChange() final;
    void isPlayingAudioWillChange() final;
    void isPlayingAudioDidChange() final;

    bool scrollingUpdatesDisabledForTesting() final;

#if ENABLE(ATTACHMENT_ELEMENT)
    void didInsertAttachment(API::Attachment&, const String& source) final;
    void didRemoveAttachment(API::Attachment&) final;
    void didInvalidateDataForAttachment(API::Attachment&) final;
    NSFileWrapper *allocFileWrapperInstance() const final;
    NSSet *serializableFileWrapperClasses() const final;
#endif

    std::optional<WebCore::DictationContext> addDictationAlternatives(PlatformTextAlternatives *) final;
    void replaceDictationAlternatives(PlatformTextAlternatives *, WebCore::DictationContext) final;
    void removeDictationAlternatives(WebCore::DictationContext) final;
    Vector<String> dictationAlternatives(WebCore::DictationContext) final;
    PlatformTextAlternatives *platformDictationAlternatives(WebCore::DictationContext) final;

#if ENABLE(APP_HIGHLIGHTS)
    void storeAppHighlight(const WebCore::AppHighlight&) final;
#endif

    void microphoneCaptureWillChange() final;
    void cameraCaptureWillChange() final;
    void displayCaptureWillChange() final;
    void displayCaptureSurfacesWillChange() final;
    void systemAudioCaptureWillChange() final;

    void microphoneCaptureChanged() final;
    void cameraCaptureChanged() final;
    void displayCaptureChanged() final;
    void displayCaptureSurfacesChanged() final;
    void systemAudioCaptureChanged() final;

    WindowKind windowKind() final;

#if ENABLE(WRITING_TOOLS)
    void proofreadingSessionShowDetailsForSuggestionWithIDRelativeToRect(const WebCore::WritingTools::TextSuggestionID&, WebCore::IntRect selectionBoundsInRootView) final;

    void proofreadingSessionUpdateStateForSuggestionWithID(WebCore::WritingTools::TextSuggestionState, const WTF::UUID& replacementUUID) final;

    void writingToolsActiveWillChange() final;
    void writingToolsActiveDidChange() final;

    void didEndPartialIntelligenceTextAnimation() final;
    bool writingToolsTextReplacementsFinished() final;

    void addTextAnimationForAnimationID(const WTF::UUID&, const WebCore::TextAnimationData&) final;
    void removeTextAnimationForAnimationID(const WTF::UUID&) final;
#endif

#if ENABLE(SCREEN_TIME)
    void installScreenTimeWebpageController() final;
    void didChangeScreenTimeWebpageControllerURL() final;
    void updateScreenTimeWebpageControllerURL(WKWebView *);
#endif

#if ENABLE(GAMEPAD)
    void setGamepadsRecentlyAccessed(GamepadsRecentlyAccessed) final;
#if PLATFORM(VISION)
    void gamepadsConnectedStateChanged() final;
#endif
#endif

    void hasActiveNowPlayingSessionChanged(bool) final;

    void videoControlsManagerDidChange() override;

    CocoaWindow *platformWindow() const final;

    void processDidUpdateThrottleState() final;

private:
#if ENABLE(FULLSCREEN_API)
    void setFullScreenClientForTesting(std::unique_ptr<WebFullScreenManagerProxyClient>&&) final;
#endif
protected:
    RetainPtr<WKWebView> webView() const { return m_webView.get(); }

    WeakObjCPtr<WKWebView> m_webView;
    std::unique_ptr<WebCore::AlternativeTextUIController> m_alternativeTextUIController;
#if ENABLE(FULLSCREEN_API)
    std::unique_ptr<WebFullScreenManagerProxyClient> m_fullscreenClientForTesting;
#endif
};

} // namespace WebKit
