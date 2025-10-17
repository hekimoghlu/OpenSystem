/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "config.h"
#include "ScrollbarThemeComposite.h"

#include "GraphicsContext.h"
#include "Scrollbar.h"

namespace WebCore {

bool ScrollbarThemeComposite::paint(Scrollbar& scrollbar, GraphicsContext& graphicsContext, const IntRect& damageRect)
{
    // Create the ScrollbarControlPartMask based on the damageRect
    ScrollbarControlPartMask scrollMask = NoPart;

    IntRect backButtonStartPaintRect;
    IntRect backButtonEndPaintRect;
    IntRect forwardButtonStartPaintRect;
    IntRect forwardButtonEndPaintRect;
    if (hasButtons(scrollbar)) {
        backButtonStartPaintRect = backButtonRect(scrollbar, BackButtonStartPart, true);
        if (damageRect.intersects(backButtonStartPaintRect))
            scrollMask |= BackButtonStartPart;
        backButtonEndPaintRect = backButtonRect(scrollbar, BackButtonEndPart, true);
        if (damageRect.intersects(backButtonEndPaintRect))
            scrollMask |= BackButtonEndPart;
        forwardButtonStartPaintRect = forwardButtonRect(scrollbar, ForwardButtonStartPart, true);
        if (damageRect.intersects(forwardButtonStartPaintRect))
            scrollMask |= ForwardButtonStartPart;
        forwardButtonEndPaintRect = forwardButtonRect(scrollbar, ForwardButtonEndPart, true);
        if (damageRect.intersects(forwardButtonEndPaintRect))
            scrollMask |= ForwardButtonEndPart;
    }

    IntRect startTrackRect;
    IntRect thumbRect;
    IntRect endTrackRect;
    IntRect trackPaintRect = trackRect(scrollbar, true);
    if (damageRect.intersects(trackPaintRect))
        scrollMask |= TrackBGPart;
    bool thumbPresent = hasThumb(scrollbar);
    if (thumbPresent) {
        IntRect track = trackRect(scrollbar);
        splitTrack(scrollbar, track, startTrackRect, thumbRect, endTrackRect);
        if (damageRect.intersects(thumbRect))
            scrollMask |= ThumbPart;
        if (damageRect.intersects(startTrackRect))
            scrollMask |= BackTrackPart;
        if (damageRect.intersects(endTrackRect))
            scrollMask |= ForwardTrackPart;
    }

    willPaintScrollbar(graphicsContext, scrollbar);
    
    // Paint the scrollbar background (only used by custom CSS scrollbars).
    paintScrollbarBackground(graphicsContext, scrollbar);

    // Paint the back and forward buttons.
    if (scrollMask & BackButtonStartPart)
        paintButton(graphicsContext, scrollbar, backButtonStartPaintRect, BackButtonStartPart);
    if (scrollMask & BackButtonEndPart)
        paintButton(graphicsContext, scrollbar, backButtonEndPaintRect, BackButtonEndPart);
    if (scrollMask & ForwardButtonStartPart)
        paintButton(graphicsContext, scrollbar, forwardButtonStartPaintRect, ForwardButtonStartPart);
    if (scrollMask & ForwardButtonEndPart)
        paintButton(graphicsContext, scrollbar, forwardButtonEndPaintRect, ForwardButtonEndPart);
    
    if (scrollMask & TrackBGPart)
        paintTrackBackground(graphicsContext, scrollbar, trackPaintRect);
    
    if ((scrollMask & ForwardTrackPart) || (scrollMask & BackTrackPart)) {
        // Paint the track pieces above and below the thumb.
        if (scrollMask & BackTrackPart)
            paintTrackPiece(graphicsContext, scrollbar, startTrackRect, BackTrackPart);
        if (scrollMask & ForwardTrackPart)
            paintTrackPiece(graphicsContext, scrollbar, endTrackRect, ForwardTrackPart);

        paintTickmarks(graphicsContext, scrollbar, trackPaintRect);
    }

    // Paint the thumb.
    if (scrollMask & ThumbPart)
        paintThumb(graphicsContext, scrollbar, thumbRect);

    didPaintScrollbar(graphicsContext, scrollbar);
    return true;
}

ScrollbarPart ScrollbarThemeComposite::hitTest(Scrollbar& scrollbar, const IntPoint& position)
{
    ScrollbarPart result = NoPart;
    if (!scrollbar.enabled())
        return result;

    IntPoint testPosition = scrollbar.convertFromContainingWindow(position);
    testPosition.move(scrollbar.x(), scrollbar.y());
    
    if (!scrollbar.frameRect().contains(testPosition))
        return NoPart;

    result = ScrollbarBGPart;

    IntRect track = trackRect(scrollbar);
    if (track.contains(testPosition)) {
        IntRect beforeThumbRect;
        IntRect thumbRect;
        IntRect afterThumbRect;
        splitTrack(scrollbar, track, beforeThumbRect, thumbRect, afterThumbRect);
        if (thumbRect.contains(testPosition))
            result = ThumbPart;
        else if (beforeThumbRect.contains(testPosition))
            result = BackTrackPart;
        else if (afterThumbRect.contains(testPosition))
            result = ForwardTrackPart;
        else
            result = TrackBGPart;
    } else if (backButtonRect(scrollbar, BackButtonStartPart).contains(testPosition))
        result = BackButtonStartPart;
    else if (backButtonRect(scrollbar, BackButtonEndPart).contains(testPosition))
        result = BackButtonEndPart;
    else if (forwardButtonRect(scrollbar, ForwardButtonStartPart).contains(testPosition))
        result = ForwardButtonStartPart;
    else if (forwardButtonRect(scrollbar, ForwardButtonEndPart).contains(testPosition))
        result = ForwardButtonEndPart;
    return result;
}

void ScrollbarThemeComposite::invalidatePart(Scrollbar& scrollbar, ScrollbarPart part)
{
    if (part == NoPart)
        return;

    IntRect result;    
    switch (part) {
        case BackButtonStartPart:
            result = backButtonRect(scrollbar, BackButtonStartPart, true);
            break;
        case BackButtonEndPart:
            result = backButtonRect(scrollbar, BackButtonEndPart, true);
            break;
        case ForwardButtonStartPart:
            result = forwardButtonRect(scrollbar, ForwardButtonStartPart, true);
            break;
        case ForwardButtonEndPart:
            result = forwardButtonRect(scrollbar, ForwardButtonEndPart, true);
            break;
        case TrackBGPart:
            result = trackRect(scrollbar, true);
            break;
        case ScrollbarBGPart:
            result = scrollbar.frameRect();
            break;
        default: {
            IntRect beforeThumbRect, thumbRect, afterThumbRect;
            splitTrack(scrollbar, trackRect(scrollbar), beforeThumbRect, thumbRect, afterThumbRect);
            if (part == BackTrackPart)
                result = beforeThumbRect;
            else if (part == ForwardTrackPart)
                result = afterThumbRect;
            else
                result = thumbRect;
        }
    }
    result.moveBy(-scrollbar.location());
    scrollbar.invalidateRect(result);
}

void ScrollbarThemeComposite::splitTrack(Scrollbar& scrollbar, const IntRect& unconstrainedTrackRect, IntRect& beforeThumbRect, IntRect& thumbRect, IntRect& afterThumbRect)
{
    // This function won't even get called unless we're big enough to have some combination of these three rects where at least
    // one of them is non-empty.
    IntRect trackRect = constrainTrackRectToTrackPieces(scrollbar, unconstrainedTrackRect);
    int thickness = scrollbar.orientation() == ScrollbarOrientation::Horizontal ? scrollbar.height() : scrollbar.width();
    int thumbPos = thumbPosition(scrollbar);
    if (scrollbar.orientation() == ScrollbarOrientation::Horizontal) {
        thumbRect = IntRect(trackRect.x() + thumbPos, trackRect.y() + (trackRect.height() - thickness) / 2, thumbLength(scrollbar), thickness);
        beforeThumbRect = IntRect(trackRect.x(), trackRect.y(), thumbPos + thumbRect.width() / 2, trackRect.height()); 
        afterThumbRect = IntRect(trackRect.x() + beforeThumbRect.width(), trackRect.y(), trackRect.maxX() - beforeThumbRect.maxX(), trackRect.height());
    } else {
        thumbRect = IntRect(trackRect.x() + (trackRect.width() - thickness) / 2, trackRect.y() + thumbPos, thickness, thumbLength(scrollbar));
        beforeThumbRect = IntRect(trackRect.x(), trackRect.y(), trackRect.width(), thumbPos + thumbRect.height() / 2); 
        afterThumbRect = IntRect(trackRect.x(), trackRect.y() + beforeThumbRect.height(), trackRect.width(), trackRect.maxY() - beforeThumbRect.maxY());
    }
}

// Returns the size represented by track taking into account scrolling past
// the end of the document.
static float usedTotalSize(Scrollbar& scrollbar)
{
    float overhangAtStart = -scrollbar.currentPos();
    float overhangAtEnd = scrollbar.currentPos() + scrollbar.visibleSize() - scrollbar.totalSize();
    float overhang = std::max(0.0f, std::max(overhangAtStart, overhangAtEnd));
    return scrollbar.totalSize() + overhang;
}

int ScrollbarThemeComposite::thumbPosition(Scrollbar& scrollbar)
{
    if (scrollbar.enabled()) {
        float size = usedTotalSize(scrollbar) - scrollbar.visibleSize();
        // Avoid doing a floating point divide by zero and return 1 when usedTotalSize == visibleSize.
        if (!size)
            return 1;
        float pos = std::max(0.0f, scrollbar.currentPos()) * (trackLength(scrollbar) - thumbLength(scrollbar)) / size;
        return (pos < 1 && pos > 0) ? 1 : pos;
    }
    return 0;
}

int ScrollbarThemeComposite::thumbLength(Scrollbar& scrollbar)
{
    if (!scrollbar.enabled())
        return 0;

    float proportion = scrollbar.visibleSize() / usedTotalSize(scrollbar);
    int trackLen = trackLength(scrollbar);
    int length = round(proportion * trackLen);
    length = std::max(length, minimumThumbLength(scrollbar));
    if (length > trackLen)
        length = 0; // Once the thumb is below the track length, it just goes away (to make more room for the track).
    return length;
}

int ScrollbarThemeComposite::minimumThumbLength(Scrollbar& scrollbar)
{
    return scrollbarThickness(scrollbar.widthStyle());
}

int ScrollbarThemeComposite::trackPosition(Scrollbar& scrollbar)
{
    IntRect constrainedTrackRect = constrainTrackRectToTrackPieces(scrollbar, trackRect(scrollbar));
    return (scrollbar.orientation() == ScrollbarOrientation::Horizontal) ? constrainedTrackRect.x() - scrollbar.x() : constrainedTrackRect.y() - scrollbar.y();
}

int ScrollbarThemeComposite::trackLength(Scrollbar& scrollbar)
{
    IntRect constrainedTrackRect = constrainTrackRectToTrackPieces(scrollbar, trackRect(scrollbar));
    return (scrollbar.orientation() == ScrollbarOrientation::Horizontal) ? constrainedTrackRect.width() : constrainedTrackRect.height();
}

IntRect ScrollbarThemeComposite::thumbRect(Scrollbar& scrollbar)
{
    if (!hasThumb(scrollbar))
        return IntRect();

    IntRect track = trackRect(scrollbar);
    IntRect startTrackRect;
    IntRect thumbRect;
    IntRect endTrackRect;
    splitTrack(scrollbar, track, startTrackRect, thumbRect, endTrackRect);

    return thumbRect;
}

void ScrollbarThemeComposite::paintOverhangAreas(ScrollView&, GraphicsContext& context, const IntRect& horizontalOverhangRect, const IntRect& verticalOverhangRect, const IntRect& dirtyRect)
{    
    context.setFillColor(Color::white);
    if (!horizontalOverhangRect.isEmpty())
        context.fillRect(intersection(horizontalOverhangRect, dirtyRect));

    context.setFillColor(Color::white);
    if (!verticalOverhangRect.isEmpty())
        context.fillRect(intersection(verticalOverhangRect, dirtyRect));
}

}
