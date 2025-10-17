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
#pragma once

#include "WebFindOptions.h"
#include "WebFoundTextRange.h"
#include <WebCore/FindOptions.h>
#include <WebCore/IntRect.h>
#include <WebCore/PageOverlay.h>
#include <WebCore/PlatformLayerIdentifier.h>
#include <WebCore/SimpleRange.h>
#include <WebCore/TextIndicator.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {
class Document;
class LocalFrame;
}

namespace WebKit {

class WebPage;

class WebFoundTextRangeController : private WebCore::PageOverlayClient {
    WTF_MAKE_TZONE_ALLOCATED(WebFoundTextRangeController);
    WTF_MAKE_NONCOPYABLE(WebFoundTextRangeController);

public:
    explicit WebFoundTextRangeController(WebPage&);

    void findTextRangesForStringMatches(const String&, OptionSet<FindOptions>, uint32_t maxMatchCount, CompletionHandler<void(Vector<WebKit::WebFoundTextRange>&&)>&&);

    void replaceFoundTextRangeWithString(const WebFoundTextRange&, const String&);

    void decorateTextRangeWithStyle(const WebFoundTextRange&, FindDecorationStyle);
    void scrollTextRangeToVisible(const WebFoundTextRange&);

    void clearAllDecoratedFoundText();

    void didBeginTextSearchOperation();

    void addLayerForFindOverlay(CompletionHandler<void(std::optional<WebCore::PlatformLayerIdentifier>)>&&);
    void removeLayerForFindOverlay();

    void requestRectForFoundTextRange(const WebFoundTextRange&, CompletionHandler<void(WebCore::FloatRect)>&&);

    void redraw();

    void clearCachedRanges();

private:
    // PageOverlayClient.
    void willMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    void didMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    bool mouseEvent(WebCore::PageOverlay&, const WebCore::PlatformMouseEvent&) override;
    void drawRect(WebCore::PageOverlay&, WebCore::GraphicsContext&, const WebCore::IntRect& dirtyRect) override;

    RefPtr<WebCore::TextIndicator> createTextIndicatorForRange(const WebCore::SimpleRange&, WebCore::TextIndicatorPresentationTransition);
    void setTextIndicatorWithRange(const WebCore::SimpleRange&);
    void flashTextIndicatorAndUpdateSelectionWithRange(const WebCore::SimpleRange&);

    RefPtr<WebCore::TextIndicator> createTextIndicatorForPDFRange(const WebFoundTextRange&, WebCore::TextIndicatorPresentationTransition);
    void setTextIndicatorWithPDFRange(const WebFoundTextRange&);
    void flashTextIndicatorAndUpdateSelectionWithPDFRange(const WebFoundTextRange&);

    Vector<WebCore::FloatRect> rectsForTextMatchesInRect(WebCore::IntRect clipRect);

    WebCore::LocalFrame* frameForFoundTextRange(const WebFoundTextRange&) const;
    WebCore::Document* documentForFoundTextRange(const WebFoundTextRange&) const;
    std::optional<WebCore::SimpleRange> simpleRangeFromFoundTextRange(WebFoundTextRange);

    WeakPtr<WebPage> m_webPage;
    RefPtr<WebCore::PageOverlay> m_findPageOverlay;

    WebFoundTextRange m_highlightedRange;

    HashMap<WebFoundTextRange, std::optional<WebCore::WeakSimpleRange>> m_cachedFoundRanges;
    HashMap<WebFoundTextRange, FindDecorationStyle> m_decoratedRanges;

    RefPtr<WebCore::TextIndicator> m_textIndicator;
};

} // namespace WebKit
