/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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

#if (ENABLE(SERVICE_CONTROLS) || ENABLE(TELEPHONE_NUMBER_DETECTION)) && PLATFORM(MAC)

#include "DataDetectorHighlight.h"
#include "GraphicsLayer.h"
#include "GraphicsLayerClient.h"
#include "PageOverlay.h"
#include "Timer.h"
#include <wtf/MonotonicTime.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {
    
class LayoutRect;
class Page;

enum class RenderingUpdateStep : uint32_t;

struct GapRects;

class ServicesOverlayController : private DataDetectorHighlightClient, private PageOverlayClient {
    WTF_MAKE_TZONE_ALLOCATED(ServicesOverlayController);
public:
    explicit ServicesOverlayController(Page&);
    ~ServicesOverlayController();

    void ref() const;
    void deref() const;

    void selectedTelephoneNumberRangesChanged();
    void selectionRectsDidChange(const Vector<LayoutRect>&, const Vector<GapRects>&, bool isTextOnly);

private:
    // PageOverlayClient
    void willMoveToPage(PageOverlay&, Page*) override;
    void didMoveToPage(PageOverlay&, Page*) override;
    void drawRect(PageOverlay&, GraphicsContext&, const IntRect& dirtyRect) override;
    bool mouseEvent(PageOverlay&, const PlatformMouseEvent&) override;
    void didScrollFrame(PageOverlay&, LocalFrame&) override;

    void createOverlayIfNeeded();
    void handleClick(const IntPoint&, DataDetectorHighlight&);

    void drawHighlight(DataDetectorHighlight&, GraphicsContext&);

    void invalidateHighlightsOfType(DataDetectorHighlight::Type);
    void buildPotentialHighlightsIfNeeded();

    void replaceHighlightsOfTypePreservingEquivalentHighlights(UncheckedKeyHashSet<RefPtr<DataDetectorHighlight>>&, DataDetectorHighlight::Type);
    void removeAllPotentialHighlightsOfType(DataDetectorHighlight::Type);
    void buildPhoneNumberHighlights();
    void buildSelectionHighlight();

    void determineActiveHighlight(bool& mouseIsOverButton);
    void clearActiveHighlight();

#if ENABLE(DATA_DETECTION)
    // DataDetectorHighlightClient
    DataDetectorHighlight* activeHighlight() const final { return m_activeHighlight.get(); }
    void scheduleRenderingUpdate(OptionSet<RenderingUpdateStep>) final;
    float deviceScaleFactor() const final;
    RefPtr<GraphicsLayer> createGraphicsLayer(GraphicsLayerClient&) final;
#endif

    DataDetectorHighlight* findTelephoneNumberHighlightContainingSelectionHighlight(DataDetectorHighlight&);

    bool hasRelevantSelectionServices();

    bool mouseIsOverHighlight(DataDetectorHighlight&, bool& mouseIsOverButton) const;
    Seconds remainingTimeUntilHighlightShouldBeShown(DataDetectorHighlight*) const;
    void determineActiveHighlightTimerFired();

    Vector<SimpleRange> telephoneNumberRangesForFocusedFrame();

    Page& page() const { return m_page; }
    Ref<Page> protectedPage() const { return m_page.get(); }

    WeakRef<Page> m_page;
    WeakPtr<PageOverlay> m_servicesOverlay;

    RefPtr<DataDetectorHighlight> m_activeHighlight;
    RefPtr<DataDetectorHighlight> m_nextActiveHighlight;
    UncheckedKeyHashSet<RefPtr<DataDetectorHighlight>> m_potentialHighlights;
    UncheckedKeyHashSet<RefPtr<DataDetectorHighlight>> m_animatingHighlights;
    WeakHashSet<DataDetectorHighlight> m_highlights;

    Vector<LayoutRect> m_currentSelectionRects;
    bool m_isTextOnly { false };

    OptionSet<DataDetectorHighlight::Type> m_dirtyHighlightTypes;

    MonotonicTime m_lastSelectionChangeTime;
    MonotonicTime m_nextActiveHighlightChangeTime;
    MonotonicTime m_lastMouseUpTime;

    RefPtr<DataDetectorHighlight> m_currentMouseDownOnButtonHighlight;
    IntPoint m_mousePosition;

    Timer m_determineActiveHighlightTimer;
    Timer m_buildHighlightsTimer;
};

} // namespace WebCore

#endif // (ENABLE(SERVICE_CONTROLS) || ENABLE(TELEPHONE_NUMBER_DETECTION)) && PLATFORM(MAC)
