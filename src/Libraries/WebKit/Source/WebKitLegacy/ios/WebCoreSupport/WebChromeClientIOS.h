/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#if PLATFORM(IOS_FAMILY)

#import "WebChromeClient.h"
#import <wtf/TZoneMalloc.h>

class WebChromeClientIOS final : public WebChromeClient {
    WTF_MAKE_TZONE_ALLOCATED(WebChromeClientIOS);
public:
    WebChromeClientIOS(WebView* webView)
        : WebChromeClient(webView)
    {
    }

private:
    void setWindowRect(const WebCore::FloatRect&) final;
    WebCore::FloatRect windowRect() const final;

    void focus() final;
    void takeFocus(WebCore::FocusDirection) final { }

    void runJavaScriptAlert(WebCore::LocalFrame&, const WTF::String&) final;
    bool runJavaScriptConfirm(WebCore::LocalFrame&, const WTF::String&) final;
    bool runJavaScriptPrompt(WebCore::LocalFrame&, const WTF::String& message, const WTF::String& defaultValue, WTF::String& result) final;

    void runOpenPanel(WebCore::LocalFrame&, WebCore::FileChooser&) final;
    void showShareSheet(WebCore::ShareDataWithParsedURL&, CompletionHandler<void(bool)>&&) final;

    bool hoverSupportedByPrimaryPointingDevice() const final { return false; }
    bool hoverSupportedByAnyAvailablePointingDevice() const final { return false; }
    std::optional<WebCore::PointerCharacteristics> pointerCharacteristicsOfPrimaryPointingDevice() const final { return WebCore::PointerCharacteristics::Coarse; }
    OptionSet<WebCore::PointerCharacteristics> pointerCharacteristicsOfAllAvailablePointingDevices() const final { return WebCore::PointerCharacteristics::Coarse; }

    void setCursor(const WebCore::Cursor&) final { }
    void setCursorHiddenUntilMouseMoves(bool) final { }

#if ENABLE(TOUCH_EVENTS)
    void didPreventDefaultForEvent() final;
#endif

    void didReceiveMobileDocType(bool) final;
    void setNeedsScrollNotifications(WebCore::LocalFrame&, bool) final;
    void didFinishContentChangeObserving(WebCore::LocalFrame&, WKContentChange) final;
    WebCore::FloatSize screenSize() const final;
    WebCore::FloatSize availableScreenSize() const final;
    WebCore::FloatSize overrideScreenSize() const final;
    WebCore::FloatSize overrideAvailableScreenSize() const final;
    void dispatchDisabledAdaptationsDidChange(const OptionSet<WebCore::DisabledAdaptations>&) const final;
    void dispatchViewportPropertiesDidChange(const WebCore::ViewportArguments&) const final;
    void notifyRevealedSelectionByScrollingFrame(WebCore::LocalFrame&) final;
    bool isStopping() final;
    void didLayout(LayoutType) final;
    void didStartOverflowScroll() final;
    void didEndOverflowScroll() final;

    void suppressFormNotifications() final;
    void restoreFormNotifications() final;

    void elementDidFocus(WebCore::Element&, const WebCore::FocusOptions&) final;
    void elementDidBlur(WebCore::Element&) final;

    void attachRootGraphicsLayer(WebCore::LocalFrame&, WebCore::GraphicsLayer*) final;

    void didFlushCompositingLayers() final;

    void updateViewportConstrainedLayers(HashMap<PlatformLayer*, std::unique_ptr<WebCore::ViewportConstraints>>&, const HashMap<PlatformLayer*, PlatformLayer*>&) final;

    bool fetchCustomFixedPositionLayoutRect(WebCore::IntRect&) final;
    void addOrUpdateScrollingLayer(WebCore::Node*, PlatformLayer*, PlatformLayer*, const WebCore::IntSize&, bool allowHorizontalScrollbar, bool allowVerticalScrollbar) final;
    void removeScrollingLayer(WebCore::Node*, PlatformLayer*, PlatformLayer*) final;

    bool selectItemWritingDirectionIsNatural() final;
    bool selectItemAlignmentFollowsMenuWritingDirection() final;
    RefPtr<WebCore::PopupMenu> createPopupMenu(WebCore::PopupMenuClient&) const final;
    RefPtr<WebCore::SearchPopupMenu> createSearchPopupMenu(WebCore::PopupMenuClient&) const final;
    void relayAccessibilityNotification(const String&, const RetainPtr<NSData>&) const final { }
    void webAppOrientationsUpdated() final;
    void focusedElementChanged(WebCore::Element*) final;
    void showPlaybackTargetPicker(bool hasVideo, WebCore::RouteSharingPolicy, const String&) final;
    RefPtr<WebCore::Icon> createIconForFiles(const Vector<String>& filenames) final;

    bool showDataDetectorsUIForElement(const WebCore::Element&, const WebCore::Event&) final { return false; }

#if ENABLE(ORIENTATION_EVENTS)
    WebCore::IntDegrees deviceOrientation() const final;
#endif

    int m_formNotificationSuppressions { 0 };
};

#endif
