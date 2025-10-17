/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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

#import "PageClientImplCocoa.h"
#import "WKBrowserEngineDefinitions.h"
#import "WebFullScreenManagerProxy.h"
#import <WebCore/InspectorOverlay.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>

OBJC_CLASS PlatformTextAlternatives;
OBJC_CLASS WKContentView;
OBJC_CLASS WKEditorUndoTarget;

namespace WebCore {
enum class DOMPasteAccessCategory : uint8_t;
enum class DOMPasteAccessResponse : uint8_t;
struct PromisedAttachmentInfo;
}

namespace WebKit {

class RemoteLayerTreeNode;

enum class ColorControlSupportsAlpha : bool;
enum class UndoOrRedo : bool;

class PageClientImpl final : public PageClientImplCocoa
#if ENABLE(FULLSCREEN_API)
    , public WebFullScreenManagerProxyClient
#endif
    {
    WTF_MAKE_FAST_ALLOCATED;
#if ENABLE(FULLSCREEN_API)
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PageClientImpl);
#endif
public:
    PageClientImpl(WKContentView *, WKWebView *);
    virtual ~PageClientImpl();
    
private:
    // PageClient
    Ref<DrawingAreaProxy> createDrawingAreaProxy(WebProcessProxy&) override;
    void setViewNeedsDisplay(const WebCore::Region&) override;
    void requestScroll(const WebCore::FloatPoint& scrollPosition, const WebCore::IntPoint& scrollOrigin, WebCore::ScrollIsAnimated) override;
    WebCore::FloatPoint viewScrollPosition() override;
    WebCore::IntSize viewSize() override;
    bool isViewWindowActive() override;
    bool isViewFocused() override;
    bool isViewVisible() override;
    void viewIsBecomingVisible() override;
    bool canTakeForegroundAssertions() override;
    bool isViewInWindow() override;
    bool isViewVisibleOrOccluded() override;
    bool isVisuallyIdle() override;
    void processDidExit() override;
    void processWillSwap() override;
    void didRelaunchProcess() override;

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
    void didCreateContextInWebProcessForVisibilityPropagation(LayerHostingContextID) override;
#if ENABLE(GPU_PROCESS)
    void didCreateContextInGPUProcessForVisibilityPropagation(LayerHostingContextID) override;
#endif // ENABLE(GPU_PROCESS)
#if ENABLE(MODEL_PROCESS)
    void didCreateContextInModelProcessForVisibilityPropagation(LayerHostingContextID) override;
#endif // ENABLE(MODEL_PROCESS)
#if USE(EXTENSIONKIT)
    UIView *createVisibilityPropagationView() override;
#endif
#endif // HAVE(VISIBILITY_PROPAGATION_VIEW)

#if ENABLE(GPU_PROCESS)
    void gpuProcessDidExit() override;
#endif
#if ENABLE(MODEL_PROCESS)
    void modelProcessDidExit() override;
#endif
    void preferencesDidChange() override;
    void toolTipChanged(const String&, const String&) override;
    void decidePolicyForGeolocationPermissionRequest(WebFrameProxy&, const FrameInfoData&, Function<void(bool)>&) override;
    void didStartProvisionalLoadForMainFrame() override;
    void didFailProvisionalLoadForMainFrame() override;
    void didCommitLoadForMainFrame(const String& mimeType, bool useCustomContentProvider) override;
    void didChangeContentSize(const WebCore::IntSize&) override;
    void setCursor(const WebCore::Cursor&) override;
    void setCursorHiddenUntilMouseMoves(bool) override;
    void registerEditCommand(Ref<WebEditCommandProxy>&&, UndoOrRedo) override;
    void clearAllEditCommands() override;
    bool canUndoRedo(UndoOrRedo) override;
    void executeUndoRedo(UndoOrRedo) override;
    void accessibilityWebProcessTokenReceived(std::span<const uint8_t>, pid_t) override;
    bool executeSavedCommandBySelector(const String& selector) override;
    void updateSecureInputState() override;
    void resetSecureInputState() override;
    void notifyInputContextAboutDiscardedComposition() override;
    void makeFirstResponder() override;
    void assistiveTechnologyMakeFirstResponder() override;
    WebCore::FloatRect convertToDeviceSpace(const WebCore::FloatRect&) override;
    WebCore::FloatRect convertToUserSpace(const WebCore::FloatRect&) override;
    WebCore::IntPoint screenToRootView(const WebCore::IntPoint&) override;
    WebCore::IntPoint rootViewToScreen(const WebCore::IntPoint&) override;
    WebCore::IntRect rootViewToScreen(const WebCore::IntRect&) override;
    WebCore::IntPoint accessibilityScreenToRootView(const WebCore::IntPoint&) override;
    WebCore::IntRect rootViewToAccessibilityScreen(const WebCore::IntRect&) override;
    void relayAccessibilityNotification(const String&, const RetainPtr<NSData>&) override;
    void doneWithKeyEvent(const NativeWebKeyboardEvent&, bool wasEventHandled) override;
#if ENABLE(TOUCH_EVENTS)
    void doneWithTouchEvent(const NativeWebTouchEvent&, bool wasEventHandled) override;
#endif
#if ENABLE(IOS_TOUCH_EVENTS)
    void doneDeferringTouchStart(bool preventNativeGestures) override;
    void doneDeferringTouchMove(bool preventNativeGestures) override;
    void doneDeferringTouchEnd(bool preventNativeGestures) override;
#endif

#if ENABLE(IMAGE_ANALYSIS)
    void requestTextRecognition(const URL& imageURL, WebCore::ShareableBitmap::Handle&& imageData, const String& sourceLanguageIdentifier, const String& targetLanguageIdentifier, CompletionHandler<void(WebCore::TextRecognitionResult&&)>&&) final;
#endif

    RefPtr<WebPopupMenuProxy> createPopupMenuProxy(WebPageProxy&) override;
    Ref<WebCore::ValidationBubble> createValidationBubble(const String& message, const WebCore::ValidationBubble::Settings&) final;

    RefPtr<WebColorPicker> createColorPicker(WebPageProxy&, const WebCore::Color& initialColor, const WebCore::IntRect&, ColorControlSupportsAlpha, Vector<WebCore::Color>&&) final { return nullptr; }

    RefPtr<WebDataListSuggestionsDropdown> createDataListSuggestionsDropdown(WebPageProxy&) final;

    WebCore::DataOwnerType dataOwnerForPasteboard(PasteboardAccessIntent) const final;

    RefPtr<WebDateTimePicker> createDateTimePicker(WebPageProxy&) final { return nullptr; }

    void setTextIndicator(Ref<WebCore::TextIndicator>, WebCore::TextIndicatorLifetime) override;
    void clearTextIndicator(WebCore::TextIndicatorDismissalAnimation) override;
    void setTextIndicatorAnimationProgress(float) override;

    void showBrowsingWarning(const BrowsingWarning&, CompletionHandler<void(std::variant<WebKit::ContinueUnsafeLoad, URL>&&)>&&) override;
    void clearBrowsingWarning() override;
    void clearBrowsingWarningIfForMainFrameNavigation() override;

    void enterAcceleratedCompositingMode(const LayerTreeContext&) override;
    void exitAcceleratedCompositingMode() override;
    void updateAcceleratedCompositingMode(const LayerTreeContext&) override;
    void setRemoteLayerTreeRootNode(RemoteLayerTreeNode*) override;
    CALayer* acceleratedCompositingRootLayer() const override;
    LayerHostingMode viewLayerHostingMode() override { return LayerHostingMode::OutOfProcess; }

    void makeViewBlank(bool) final;

    RefPtr<ViewSnapshot> takeViewSnapshot(std::optional<WebCore::IntRect>&&) override;
    void wheelEventWasNotHandledByWebCore(const NativeWebWheelEvent&) override;

    void commitPotentialTapFailed() override;
    void didGetTapHighlightGeometries(WebKit::TapIdentifier requestID, const WebCore::Color&, const Vector<WebCore::FloatQuad>& highlightedQuads, const WebCore::IntSize& topLeftRadius, const WebCore::IntSize& topRightRadius, const WebCore::IntSize& bottomLeftRadius, const WebCore::IntSize& bottomRightRadius, bool nodeHasBuiltInClickHandling) override;

    void didCommitLayerTree(const RemoteLayerTreeTransaction&) override;
    void layerTreeCommitComplete() override;
        
    void didPerformDictionaryLookup(const WebCore::DictionaryPopupInfo&) override;

    bool effectiveAppearanceIsDark() const override;
    bool effectiveUserInterfaceLevelIsElevated() const override;

    void couldNotRestorePageState() override;
    void restorePageState(std::optional<WebCore::FloatPoint>, const WebCore::FloatPoint&, const WebCore::FloatBoxExtent&, double) override;
    void restorePageCenterAndScale(std::optional<WebCore::FloatPoint>, double) override;

    void elementDidFocus(const FocusedElementInformation&, bool userIsInteracting, bool blurPreviousNode, OptionSet<WebCore::ActivityState> activityStateChanges, API::Object* userData) override;
    void updateInputContextAfterBlurringAndRefocusingElement() final;
    void didProgrammaticallyClearFocusedElement(WebCore::ElementContext&&) final;
    void updateFocusedElementInformation(const FocusedElementInformation&) final;
    void elementDidBlur() override;
    void focusedElementDidChangeInputMode(WebCore::InputMode) override;
    void didUpdateEditorState() override;
    void reconcileEnclosingScrollViewContentOffset(EditorState&) final;
    bool isFocusingElement() override;
    void selectionDidChange() override;
    bool interpretKeyEvent(const NativeWebKeyboardEvent&, bool isCharEvent) override;
    void positionInformationDidChange(const InteractionInformationAtPosition&) override;
    void saveImageToLibrary(Ref<WebCore::SharedBuffer>&&) override;
    void showPlaybackTargetPicker(bool hasVideo, const WebCore::IntRect& elementRect, WebCore::RouteSharingPolicy, const String&) override;
    void showDataDetectorsUIForPositionInformation(const InteractionInformationAtPosition&) override;

    void hardwareKeyboardAvailabilityChanged() override;

    bool handleRunOpenPanel(WebPageProxy*, WebFrameProxy*, const FrameInfoData&, API::OpenPanelParameters*, WebOpenPanelResultListenerProxy*) override;
    bool showShareSheet(const WebCore::ShareDataWithParsedURL&, WTF::CompletionHandler<void(bool)>&&) override;
    void showContactPicker(const WebCore::ContactsRequestData&, WTF::CompletionHandler<void(std::optional<Vector<WebCore::ContactInfo>>&&)>&&) override;
    
    void disableDoubleTapGesturesDuringTapIfNecessary(WebKit::TapIdentifier) override;
    void handleSmartMagnificationInformationForPotentialTap(WebKit::TapIdentifier, const WebCore::FloatRect& renderRect, bool fitEntireRect, double viewportMinimumScale, double viewportMaximumScale, bool nodeIsRootLevel) override;

    double minimumZoomScale() const override;
    WebCore::FloatRect documentRect() const override;

    void showInspectorHighlight(const WebCore::InspectorOverlay::Highlight&) override;
    void hideInspectorHighlight() override;

    void showInspectorIndication() override;
    void hideInspectorIndication() override;

    void enableInspectorNodeSearch() override;
    void disableInspectorNodeSearch() override;

    void scrollingNodeScrollViewWillStartPanGesture(WebCore::ScrollingNodeID) override;
    void scrollingNodeScrollViewDidScroll(WebCore::ScrollingNodeID) override;
    void scrollingNodeScrollWillStartScroll(std::optional<WebCore::ScrollingNodeID>) override;
    void scrollingNodeScrollDidEndScroll(std::optional<WebCore::ScrollingNodeID>) override;
        
    void requestScrollToRect(const WebCore::FloatRect& targetRect, const WebCore::FloatPoint& origin) override;

#if ENABLE(FULLSCREEN_API)
    WebFullScreenManagerProxyClient& fullScreenManagerProxyClient() override;

    // WebFullScreenManagerProxyClient
    void closeFullScreenManager() override;
    bool isFullScreen() override;
    void enterFullScreen(WebCore::FloatSize mediaDimensions, CompletionHandler<void(bool)>&&) override;
#if ENABLE(QUICKLOOK_FULLSCREEN)
    void updateImageSource() override;
#endif
    void exitFullScreen() override;
    void beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
    void beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
    bool lockFullscreenOrientation(WebCore::ScreenOrientationType) override;
    void unlockFullscreenOrientation() override;
#endif

    void didFinishLoadingDataForCustomContentProvider(const String& suggestedFilename, std::span<const uint8_t>) override;

    Vector<String> mimeTypesWithCustomContentProviders() override;

    void navigationGestureDidBegin() override;
    void navigationGestureWillEnd(bool willNavigate, WebBackForwardListItem&) override;
    void navigationGestureDidEnd(bool willNavigate, WebBackForwardListItem&) override;
    void navigationGestureDidEnd() override;
    void willRecordNavigationSnapshot(WebBackForwardListItem&) override;
    void didRemoveNavigationGestureSnapshot() override;

    void didFirstVisuallyNonEmptyLayoutForMainFrame() override;
    void didFinishNavigation(API::Navigation*) override;
    void didFailNavigation(API::Navigation*) override;
    void didSameDocumentNavigationForMainFrame(SameDocumentNavigationType) override;
    void didNotHandleTapAsClick(const WebCore::IntPoint&) override;
    void didHandleTapAsHover() override;
    void didCompleteSyntheticClick() override;

    void runModalJavaScriptDialog(CompletionHandler<void()>&& callback) final;

    void didChangeBackgroundColor() override;
    void videoControlsManagerDidChange() override;

    void refView() override;
    void derefView() override;

    void didRestoreScrollPosition() override;

    WebCore::UserInterfaceLayoutDirection userInterfaceLayoutDirection() override;

#if USE(QUICK_LOOK)
    void requestPasswordForQuickLookDocument(const String& fileName, WTF::Function<void(const String&)>&&) override;
#endif

    void requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::DOMPasteRequiresInteraction, const WebCore::IntRect& elementRect, const String&, CompletionHandler<void(WebCore::DOMPasteAccessResponse)>&&) final;

#if ENABLE(DRAG_SUPPORT)
    void didPerformDragOperation(bool handled) override;
    void didHandleDragStartRequest(bool started) override;
    void didHandleAdditionalDragItemsRequest(bool added) override;
    void startDrag(const WebCore::DragItem&, WebCore::ShareableBitmap::Handle&& image) override;
    void willReceiveEditDragSnapshot() override;
    void didReceiveEditDragSnapshot(std::optional<WebCore::TextIndicatorData>) override;
    void didChangeDragCaretRect(const WebCore::IntRect& previousCaretRect, const WebCore::IntRect& caretRect) override;
#endif

    void performSwitchHapticFeedback() final;

    void handleAutocorrectionContext(const WebAutocorrectionContext&) final;

    void setMouseEventPolicy(WebCore::MouseEventPolicy) final;

#if ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS) && USE(UICONTEXTMENU)
    void showMediaControlsContextMenu(WebCore::FloatRect&&, Vector<WebCore::MediaControlsContextMenuItem>&&, CompletionHandler<void(WebCore::MediaControlsContextMenuItem::ID)>&&) final;
#endif // ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS) && USE(UICONTEXTMENU)

#if ENABLE(ATTACHMENT_ELEMENT)
    void writePromisedAttachmentToPasteboard(WebCore::PromisedAttachmentInfo&&) final;
#endif

    void cancelPointersForGestureRecognizer(UIGestureRecognizer*) override;
    std::optional<unsigned> activeTouchIdentifierForGestureRecognizer(UIGestureRecognizer*) override;

    void showDictationAlternativeUI(const WebCore::FloatRect&, WebCore::DictationContext) final;

#if HAVE(UISCROLLVIEW_ASYNCHRONOUS_SCROLL_EVENT_HANDLING)
    void handleAsynchronousCancelableScrollEvent(WKBaseScrollView *, WKBEScrollViewScrollUpdate *, void (^completion)(BOOL handled)) final;
#endif

    bool isSimulatingCompatibilityPointerTouches() const final;

    WebCore::Color contentViewBackgroundColor() final;
    WebCore::Color insertionPointColor() final;
    bool isScreenBeingCaptured() final;

    String sceneID() final;

    void beginTextRecognitionForFullscreenVideo(WebCore::ShareableBitmap::Handle&&, AVPlayerViewController *) final;
    void cancelTextRecognitionForFullscreenVideo(AVPlayerViewController *) final;
    bool isTextRecognitionInFullscreenVideoEnabled() const final;

#if ENABLE(IMAGE_ANALYSIS) && ENABLE(VIDEO)
    void beginTextRecognitionForVideoInElementFullscreen(WebCore::ShareableBitmap::Handle&&, WebCore::FloatRect) final;
    void cancelTextRecognitionForVideoInElementFullscreen() final;
#endif

    bool hasResizableWindows() const final;

#if ENABLE(VIDEO_PRESENTATION_MODE)
    void didEnterFullscreen() final { };
    void didExitFullscreen() final { };
    void didCleanupFullscreen() final;
#endif

    UIViewController *presentingViewController() const final;

    bool isPotentialTapInProgress() const final;

    WebCore::FloatPoint webViewToRootView(const WebCore::FloatPoint&) const final;
    WebCore::FloatRect rootViewToWebView(const WebCore::FloatRect&) const final;

#if HAVE(SPATIAL_TRACKING_LABEL)
    const String& spatialTrackingLabel() const final;
#endif

#if ENABLE(PDF_PLUGIN)
    void pluginDidInstallPDFDocument(double initialScale) final;
#endif

    void scheduleVisibleContentRectUpdate() final;

    RetainPtr<WKContentView> contentView() const { return m_contentView.get(); }

    WeakObjCPtr<WKContentView> m_contentView;
    RetainPtr<WKEditorUndoTarget> m_undoTarget;
};
} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
