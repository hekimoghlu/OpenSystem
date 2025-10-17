/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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

#include "WebFindOptions.h"
#include <WebCore/FindOptions.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/IntRect.h>
#include <WebCore/PageOverlay.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/SimpleRange.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

#if PLATFORM(IOS_FAMILY)
#include "FindIndicatorOverlayClientIOS.h"
#endif

namespace WebCore {
class LocalFrame;
class Range;
enum class DidWrap : bool;
}

namespace WebKit {

class CallbackID;
class PluginView;
class WebPage;

class FindController final : private WebCore::PageOverlayClient {
    WTF_MAKE_TZONE_ALLOCATED(FindController);
    WTF_MAKE_NONCOPYABLE(FindController);

public:
    enum class TriggerImageAnalysis : bool { No, Yes };

    explicit FindController(WebPage*);
    virtual ~FindController();

    void findString(const String&, OptionSet<FindOptions>, unsigned maxMatchCount, CompletionHandler<void(std::optional<WebCore::FrameIdentifier>, Vector<WebCore::IntRect>&&, uint32_t, int32_t, bool)>&&);
#if ENABLE(IMAGE_ANALYSIS)
    void findStringIncludingImages(const String&, OptionSet<FindOptions>, unsigned maxMatchCount, CompletionHandler<void(std::optional<WebCore::FrameIdentifier>, Vector<WebCore::IntRect>&&, uint32_t, int32_t, bool)>&&);
#endif
    void findStringMatches(const String&, OptionSet<FindOptions>, unsigned maxMatchCount, CompletionHandler<void(Vector<Vector<WebCore::IntRect>>, int32_t)>&&);
    void findRectsForStringMatches(const String&, OptionSet<WebKit::FindOptions>, unsigned maxMatchCount, CompletionHandler<void(Vector<WebCore::FloatRect>&&)>&&);
    void getImageForFindMatch(uint32_t matchIndex);
    void selectFindMatch(uint32_t matchIndex);
    void indicateFindMatch(uint32_t matchIndex);
    void hideFindUI();
    void countStringMatches(const String&, OptionSet<FindOptions>, unsigned maxMatchCount, CompletionHandler<void(uint32_t)>&&);
    uint32_t replaceMatches(const Vector<uint32_t>& matchIndices, const String& replacementText, bool selectionOnly);
    
    void hideFindIndicator();
    void resetMatchIndex();
    void showFindIndicatorInSelection();

    bool isShowingOverlay() const { return m_isShowingFindIndicator && m_findPageOverlay; }

    void deviceScaleFactorDidChange();
    void didInvalidateFindRects();

    void redraw();

private:
    // PageOverlayClient.
    void willMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    void didMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    bool mouseEvent(WebCore::PageOverlay&, const WebCore::PlatformMouseEvent&) override;
    void drawRect(WebCore::PageOverlay&, WebCore::GraphicsContext&, const WebCore::IntRect& dirtyRect) override;

    Vector<WebCore::FloatRect> rectsForTextMatchesInRect(WebCore::IntRect clipRect);
    bool updateFindIndicator(bool isShowingOverlay, bool shouldAnimate = true);

    void updateFindUIAfterPageScroll(bool found, const String&, OptionSet<FindOptions>, unsigned maxMatchCount, WebCore::DidWrap, std::optional<WebCore::FrameIdentifier>, CompletionHandler<void(std::optional<WebCore::FrameIdentifier>, Vector<WebCore::IntRect>&&, uint32_t, int32_t, bool)>&& = [](auto&&...) { });

    void willFindString();
    void didFindString();
    void didFailToFindString();
    void didHideFindIndicator();
    
    unsigned findIndicatorRadius() const;
    bool shouldHideFindIndicatorOnScroll() const;
    void didScrollAffectingFindIndicatorPosition();

    RefPtr<WebCore::LocalFrame> frameWithSelection(WebCore::Page*);

#if ENABLE(PDF_PLUGIN)
    PluginView* mainFramePlugIn();
#endif

    WeakPtr<WebPage> m_webPage;
    WeakPtr<WebCore::PageOverlay> m_findPageOverlay;

    // Whether the UI process is showing the find indicator. Note that this can be true even if
    // the find indicator isn't showing, but it will never be false when it is showing.
    bool m_isShowingFindIndicator { false };
    WebCore::IntRect m_findIndicatorRect;
    Vector<WebCore::SimpleRange> m_findMatches;
    // Index value is -1 if not found or if number of matches exceeds provided maximum.
    int m_foundStringMatchIndex { -1 };

#if PLATFORM(IOS_FAMILY)
    RefPtr<WebCore::PageOverlay> m_findIndicatorOverlay;
    std::unique_ptr<FindIndicatorOverlayClientIOS> m_findIndicatorOverlayClient;
#endif
};

} // namespace WebKit
