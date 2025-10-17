/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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

#include "GraphicsContext.h"
#include "IntRect.h"
#include "ScrollTypes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PlatformMouseEvent;
class ScrollableArea;
class Scrollbar;
class ScrollView;

#if HAVE(RUBBER_BANDING)
class GraphicsLayer;
#endif

class ScrollbarTheme {
    WTF_MAKE_TZONE_ALLOCATED(ScrollbarTheme);
    WTF_MAKE_NONCOPYABLE(ScrollbarTheme);
public:
    ScrollbarTheme() = default;
    virtual ~ScrollbarTheme() {};

    virtual void updateEnabledState(Scrollbar&) { }

    virtual bool paint(Scrollbar&, GraphicsContext&, const IntRect& /*damageRect*/) { return false; }
    virtual ScrollbarPart hitTest(Scrollbar&, const IntPoint&) { return NoPart; }
    
    virtual int scrollbarThickness(ScrollbarWidth = ScrollbarWidth::Auto, ScrollbarExpansionState = ScrollbarExpansionState::Expanded, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IncludeOverlayScrollbarSize) { return 0; }

    virtual ScrollbarButtonsPlacement buttonsPlacement() const { return ScrollbarButtonsSingle; }

    virtual bool supportsControlTints() const { return false; }
    virtual bool usesOverlayScrollbars() const { return false; }
    virtual void usesOverlayScrollbarsChanged() { }
    virtual void updateScrollbarOverlayStyle(Scrollbar&) { }

    virtual void themeChanged() {}
    
    virtual bool invalidateOnMouseEnterExit() { return false; }

    void invalidateParts(Scrollbar& scrollbar, ScrollbarControlPartMask mask)
    {
        if (mask & BackButtonStartPart)
            invalidatePart(scrollbar, BackButtonStartPart);
        if (mask & ForwardButtonStartPart)
            invalidatePart(scrollbar, ForwardButtonStartPart);
        if (mask & BackTrackPart)
            invalidatePart(scrollbar, BackTrackPart);
        if (mask & ThumbPart)
            invalidatePart(scrollbar, ThumbPart);
        if (mask & ForwardTrackPart)
            invalidatePart(scrollbar, ForwardTrackPart);
        if (mask & BackButtonEndPart)
            invalidatePart(scrollbar, BackButtonEndPart);
        if (mask & ForwardButtonEndPart)
            invalidatePart(scrollbar, ForwardButtonEndPart);
    }

    virtual void invalidatePart(Scrollbar&, ScrollbarPart) { }

    virtual void paintScrollCorner(ScrollableArea& area, GraphicsContext& context, const IntRect& cornerRect) { defaultPaintScrollCorner(area, context, cornerRect); }
    static void defaultPaintScrollCorner(ScrollableArea&, GraphicsContext& context, const IntRect& cornerRect) { context.fillRect(cornerRect, Color::white); }

    virtual void paintTickmarks(GraphicsContext&, Scrollbar&, const IntRect&) { }
    virtual void paintOverhangAreas(ScrollView&, GraphicsContext&, const IntRect&, const IntRect&, const IntRect&) { }

    virtual ScrollbarButtonPressAction handleMousePressEvent(Scrollbar&, const PlatformMouseEvent&, ScrollbarPart);
    virtual bool shouldSnapBackToDragOrigin(Scrollbar&, const PlatformMouseEvent&) { return false; }
    virtual bool shouldDragDocumentInsteadOfThumb(Scrollbar&, const PlatformMouseEvent&) { return false; }
    virtual int thumbPosition(Scrollbar&) { return 0; } // The position of the thumb relative to the track.
    virtual int thumbLength(Scrollbar&) { return 0; } // The length of the thumb along the axis of the scrollbar.
    virtual int trackPosition(Scrollbar&) { return 0; } // The position of the track relative to the scrollbar.
    virtual int trackLength(Scrollbar&) { return 0; } // The length of the track along the axis of the scrollbar.

    virtual int maxOverlapBetweenPages() { return std::numeric_limits<int>::max(); }

    virtual Seconds initialAutoscrollTimerDelay() { return 250_ms; }
    virtual Seconds autoscrollTimerDelay() { return 50_ms; }

    virtual void registerScrollbar(Scrollbar&) { }
    virtual void unregisterScrollbar(Scrollbar&) { }
    virtual void didCreateScrollerImp(Scrollbar&) { };

    virtual bool isMockTheme() const { return false; }

    WEBCORE_EXPORT static ScrollbarTheme& theme();

private:
    static ScrollbarTheme& nativeTheme(); // Must be implemented to return the correct theme subclass.
};

}
