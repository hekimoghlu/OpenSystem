/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#import <WebCore/ChromeClient.h>
#import <WebCore/FocusDirection.h>
#import <wtf/Forward.h>
#import <wtf/TZoneMalloc.h>

namespace WebCore {
class HTMLImageElement;
}

@class WebView;

// FIXME: This class is used as a concrete class on Mac, but on iOS this is an abstract
// base class of the concrete class, WebChromeClientIOS. Because of that, this class and
// many of its functions are not marked final. That is messy way to organize things.
class WebChromeClient : public WebCore::ChromeClient {
    WTF_MAKE_TZONE_ALLOCATED(WebChromeClient);
public:
    WebChromeClient(WebView*);

    WebView* webView() const { return m_webView; }

private:
    void chromeDestroyed() final;

    void setWindowRect(const WebCore::FloatRect&) override;
    WebCore::FloatRect windowRect() const override;

    WebCore::FloatRect pageRect() const final;

    void focus() override;
    void unfocus() final;

    bool canTakeFocus(WebCore::FocusDirection) const final;
    void takeFocus(WebCore::FocusDirection) override;

    void focusedElementChanged(WebCore::Element*) override;
    void focusedFrameChanged(WebCore::Frame*) final;

    RefPtr<WebCore::Page> createWindow(WebCore::LocalFrame&, const String& openedMainFrameName, const WebCore::WindowFeatures&, const WebCore::NavigationAction&) final;
    void show() final;

    bool canRunModal() const final;
    void runModal() final;

    void setToolbarsVisible(bool) final;
    bool toolbarsVisible() const final;

    void setStatusbarVisible(bool) final;
    bool statusbarVisible() const final;

    void setScrollbarsVisible(bool) final;
    bool scrollbarsVisible() const final;

    void setMenubarVisible(bool) final;
    bool menubarVisible() const final;

    void setResizable(bool) final;

    void addMessageToConsole(JSC::MessageSource, JSC::MessageLevel, const String& message, unsigned lineNumber, unsigned columnNumber, const String& sourceURL) final;

    bool canRunBeforeUnloadConfirmPanel() final;
    bool runBeforeUnloadConfirmPanel(const String& message, WebCore::LocalFrame&) final;

    void closeWindow() final;

    void rootFrameAdded(const WebCore::LocalFrame&) final { }
    void rootFrameRemoved(const WebCore::LocalFrame&) final { }

    void runJavaScriptAlert(WebCore::LocalFrame&, const String&) override;
    bool runJavaScriptConfirm(WebCore::LocalFrame&, const String&) override;
    bool runJavaScriptPrompt(WebCore::LocalFrame&, const String& message, const String& defaultValue, String& result) override;

    void invalidateRootView(const WebCore::IntRect&) final;
    void invalidateContentsAndRootView(const WebCore::IntRect&) final;
    void invalidateContentsForSlowScroll(const WebCore::IntRect&) final;
    void scroll(const WebCore::IntSize& scrollDelta, const WebCore::IntRect& rectToScroll, const WebCore::IntRect& clipRect) final;

    WebCore::IntPoint screenToRootView(const WebCore::IntPoint&) const final;
    WebCore::IntPoint rootViewToScreen(const WebCore::IntPoint&) const final;
    WebCore::IntRect rootViewToScreen(const WebCore::IntRect&) const final;

    WebCore::IntPoint accessibilityScreenToRootView(const WebCore::IntPoint&) const final;
    WebCore::IntRect rootViewToAccessibilityScreen(const WebCore::IntRect&) const final;

    void didFinishLoadingImageForElement(WebCore::HTMLImageElement&) final;

    PlatformPageClient platformPageClient() const final;
    void contentsSizeChanged(WebCore::LocalFrame&, const WebCore::IntSize&) const final;
    void intrinsicContentsSizeChanged(const WebCore::IntSize&) const final { }

    void scrollContainingScrollViewsToRevealRect(const WebCore::IntRect&) const final;

    bool shouldUnavailablePluginMessageBeButton(WebCore::PluginUnavailabilityReason) const final;
    void unavailablePluginButtonClicked(WebCore::Element&, WebCore::PluginUnavailabilityReason) const final;
    void mouseDidMoveOverElement(const WebCore::HitTestResult&, OptionSet<WebCore::PlatformEventModifier>, const String&, WebCore::TextDirection) final;

    void setToolTip(const String&);

    void print(WebCore::LocalFrame&, const WebCore::StringWithDirection&) final;
    void exceededDatabaseQuota(WebCore::LocalFrame&, const String& databaseName, WebCore::DatabaseDetails) final;

    void runOpenPanel(WebCore::LocalFrame&, WebCore::FileChooser&) override;
    void showShareSheet(WebCore::ShareDataWithParsedURL&, CompletionHandler<void(bool)>&&) override;

    void loadIconForFiles(const Vector<String>&, WebCore::FileIconLoader&) final;
    RefPtr<WebCore::Icon> createIconForFiles(const Vector<String>& filenames) override;

#if !PLATFORM(IOS_FAMILY)
    void setCursor(const WebCore::Cursor&) final;
    void setCursorHiddenUntilMouseMoves(bool) final;
#endif

    RefPtr<WebCore::ColorChooser> createColorChooser(WebCore::ColorChooserClient&, const WebCore::Color&) final;

    RefPtr<WebCore::DataListSuggestionPicker> createDataListSuggestionPicker(WebCore::DataListSuggestionsClient&) final;
    bool canShowDataListSuggestionLabels() const final { return false; }

    RefPtr<WebCore::DateTimeChooser> createDateTimeChooser(WebCore::DateTimeChooserClient&) final;

    void setTextIndicator(const WebCore::TextIndicatorData&) const final;

#if ENABLE(POINTER_LOCK)
    bool requestPointerLock() final;
    void requestPointerUnlock() final;
#endif

    WebCore::KeyboardUIMode keyboardUIMode() final;

    bool hoverSupportedByPrimaryPointingDevice() const override { return true; }
    bool hoverSupportedByAnyAvailablePointingDevice() const override { return true; }
    std::optional<WebCore::PointerCharacteristics> pointerCharacteristicsOfPrimaryPointingDevice() const override { return WebCore::PointerCharacteristics::Fine; }
    OptionSet<WebCore::PointerCharacteristics> pointerCharacteristicsOfAllAvailablePointingDevices() const override { return WebCore::PointerCharacteristics::Fine; }

    NSResponder *firstResponder() final;
    void makeFirstResponder(NSResponder *) final;

    void enableSuddenTermination() final;
    void disableSuddenTermination() final;

#if !PLATFORM(IOS_FAMILY)
    void elementDidFocus(WebCore::Element&, const WebCore::FocusOptions&) override;
    void elementDidBlur(WebCore::Element&) override;
#endif

    bool shouldPaintEntireContents() const final;

    void attachRootGraphicsLayer(WebCore::LocalFrame&, WebCore::GraphicsLayer*) override;
    void attachViewOverlayGraphicsLayer(WebCore::GraphicsLayer*) final;
    void setNeedsOneShotDrawingSynchronization() final;
    void triggerRenderingUpdate() final;

    CompositingTriggerFlags allowedCompositingTriggers() const final
    {
        return static_cast<CompositingTriggerFlags>(
            ThreeDTransformTrigger |
            VideoTrigger |
            PluginTrigger| 
            CanvasTrigger |
#if PLATFORM(IOS_FAMILY)
            AnimatedOpacityTrigger | // Allow opacity animations to trigger compositing mode for iOS: <rdar://problem/7830677>
#endif
            AnimationTrigger);
    }

#if ENABLE(VIDEO) && PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)
    void setUpPlaybackControlsManager(WebCore::HTMLMediaElement&) final;
    void clearPlaybackControlsManager() final;
    void mediaEngineChanged(WebCore::HTMLMediaElement&) final;
#endif

#if ENABLE(VIDEO)
    bool canEnterVideoFullscreen(WebCore::HTMLMediaElementEnums::VideoFullscreenMode) const final;
    bool supportsVideoFullscreen(WebCore::HTMLMediaElementEnums::VideoFullscreenMode) final;
#if ENABLE(VIDEO_PRESENTATION_MODE)
    void setMockVideoPresentationModeEnabled(bool) final;
    void enterVideoFullscreenForVideoElement(WebCore::HTMLVideoElement&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode, bool standby) final;
    void exitVideoFullscreenForVideoElement(WebCore::HTMLVideoElement&, WTF::CompletionHandler<void(bool)>&& = [](bool) { }) final;
    void exitVideoFullscreenToModeWithoutAnimation(WebCore::HTMLVideoElement&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode) final;
#endif
#endif

#if ENABLE(FULLSCREEN_API)
    bool supportsFullScreenForElement(const WebCore::Element&, bool withKeyboard) final;
    void enterFullScreenForElement(WebCore::Element&, WebCore::HTMLMediaElementEnums::VideoFullscreenMode = WebCore::HTMLMediaElementEnums::VideoFullscreenModeStandard) final;
    void exitFullScreenForElement(WebCore::Element*) final;
#endif

    bool selectItemWritingDirectionIsNatural() override;
    bool selectItemAlignmentFollowsMenuWritingDirection() override;
    RefPtr<WebCore::PopupMenu> createPopupMenu(WebCore::PopupMenuClient&) const override;
    RefPtr<WebCore::SearchPopupMenu> createSearchPopupMenu(WebCore::PopupMenuClient&) const override;

    void wheelEventHandlersChanged(bool) final { }

#if ENABLE(SERVICE_CONTROLS)
    void handleSelectionServiceClick(WebCore::FrameSelection&, const Vector<String>& telephoneNumbers, const WebCore::IntPoint&) final;
    bool hasRelevantSelectionServices(bool isTextOnly) const final;
#endif

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)
    void addPlaybackTargetPickerClient(WebCore::PlaybackTargetClientContextIdentifier) final;
    void removePlaybackTargetPickerClient(WebCore::PlaybackTargetClientContextIdentifier) final;
    void showPlaybackTargetPicker(WebCore::PlaybackTargetClientContextIdentifier, const WebCore::IntPoint&, bool /* hasVideo */) final;
    void playbackTargetPickerClientStateDidChange(WebCore::PlaybackTargetClientContextIdentifier, WebCore::MediaProducerMediaStateFlags) final;
    void setMockMediaPlaybackTargetPickerEnabled(bool) final;
    void setMockMediaPlaybackTargetPickerState(const String&, WebCore::MediaPlaybackTargetContext::MockState) final;
    void mockMediaPlaybackTargetPickerDismissPopup() override;
#endif

#if PLATFORM(MAC)
    void changeUniversalAccessZoomFocus(const WebCore::IntRect&, const WebCore::IntRect&) final;
#endif
#if HAVE(WEBGPU_IMPLEMENTATION)
    RefPtr<WebCore::WebGPU::GPU> createGPUForWebGPU() const final;
#endif
    RefPtr<WebCore::ShapeDetection::BarcodeDetector> createBarcodeDetector(const WebCore::ShapeDetection::BarcodeDetectorOptions&) const final;
    void getBarcodeDetectorSupportedFormats(CompletionHandler<void(Vector<WebCore::ShapeDetection::BarcodeFormat>&&)>&&) const final;
    RefPtr<WebCore::ShapeDetection::FaceDetector> createFaceDetector(const WebCore::ShapeDetection::FaceDetectorOptions&) const final;
    RefPtr<WebCore::ShapeDetection::TextDetector> createTextDetector() const final;

    void registerBlobPathForTesting(const String&, CompletionHandler<void()>&&) final;

    void requestCookieConsent(CompletionHandler<void(WebCore::CookieConsentDecisionResult)>&&) final;

#if ENABLE(VIDEO_PRESENTATION_MODE)
    bool m_mockVideoPresentationModeEnabled { false };
#endif

    __weak WebView *m_webView;
};
