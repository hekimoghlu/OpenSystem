/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#include "RenderScrollbarTheme.h"
#include "RenderScrollbar.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

RenderScrollbarTheme* RenderScrollbarTheme::renderScrollbarTheme()
{
    static NeverDestroyed<RenderScrollbarTheme> theme;
    return &theme.get();
}

void RenderScrollbarTheme::buttonSizesAlongTrackAxis(Scrollbar& scrollbar, int& beforeSize, int& afterSize)
{
    IntRect firstButton = backButtonRect(scrollbar, BackButtonStartPart);
    IntRect secondButton = forwardButtonRect(scrollbar, ForwardButtonStartPart);
    IntRect thirdButton = backButtonRect(scrollbar, BackButtonEndPart);
    IntRect fourthButton = forwardButtonRect(scrollbar, ForwardButtonEndPart);
    if (scrollbar.orientation() == ScrollbarOrientation::Horizontal) {
        beforeSize = firstButton.width() + secondButton.width();
        afterSize = thirdButton.width() + fourthButton.width();
    } else {
        beforeSize = firstButton.height() + secondButton.height();
        afterSize = thirdButton.height() + fourthButton.height();
    }
}

bool RenderScrollbarTheme::hasButtons(Scrollbar& scrollbar)
{
    int startSize;
    int endSize;
    buttonSizesAlongTrackAxis(scrollbar, startSize, endSize);
    return (startSize + endSize) <= (scrollbar.orientation() == ScrollbarOrientation::Horizontal ? scrollbar.width() : scrollbar.height());
}

bool RenderScrollbarTheme::hasThumb(Scrollbar& scrollbar)
{
    return trackLength(scrollbar) - thumbLength(scrollbar) >= 0;
}

int RenderScrollbarTheme::minimumThumbLength(Scrollbar& scrollbar)
{
    return downcast<RenderScrollbar>(scrollbar).minimumThumbLength();
}

IntRect RenderScrollbarTheme::backButtonRect(Scrollbar& scrollbar, ScrollbarPart partType, bool)
{
    return downcast<RenderScrollbar>(scrollbar).buttonRect(partType);
}

IntRect RenderScrollbarTheme::forwardButtonRect(Scrollbar& scrollbar, ScrollbarPart partType, bool)
{
    return downcast<RenderScrollbar>(scrollbar).buttonRect(partType);
}

IntRect RenderScrollbarTheme::trackRect(Scrollbar& scrollbar, bool)
{
    if (!hasButtons(scrollbar))
        return scrollbar.frameRect();
    
    int startLength;
    int endLength;
    buttonSizesAlongTrackAxis(scrollbar, startLength, endLength);
    
    return downcast<RenderScrollbar>(scrollbar).trackRect(startLength, endLength);
}

IntRect RenderScrollbarTheme::constrainTrackRectToTrackPieces(Scrollbar& scrollbar, const IntRect& rect)
{ 
    IntRect backRect = downcast<RenderScrollbar>(scrollbar).trackPieceRectWithMargins(BackTrackPart, rect);
    IntRect forwardRect = downcast<RenderScrollbar>(scrollbar).trackPieceRectWithMargins(ForwardTrackPart, rect);
    IntRect result = rect;
    if (scrollbar.orientation() == ScrollbarOrientation::Horizontal) {
        result.setX(backRect.x());
        result.setWidth(forwardRect.maxX() - backRect.x());
    } else {
        result.setY(backRect.y());
        result.setHeight(forwardRect.maxY() - backRect.y());
    }
    return result;
}

void RenderScrollbarTheme::willPaintScrollbar(GraphicsContext& context, Scrollbar& scrollbar)
{
    float opacity = downcast<RenderScrollbar>(scrollbar).opacity();
    if (opacity != 1) {
        context.save();
        context.clip(scrollbar.frameRect());
        context.beginTransparencyLayer(opacity);
    }
}

void RenderScrollbarTheme::didPaintScrollbar(GraphicsContext& context, Scrollbar& scrollbar)
{
    float opacity = downcast<RenderScrollbar>(scrollbar).opacity();
    if (opacity != 1) {
        context.endTransparencyLayer();
        context.restore();
    }
}

void RenderScrollbarTheme::paintScrollCorner(ScrollableArea& area, GraphicsContext& context, const IntRect& cornerRect)
{
    ScrollbarTheme::theme().paintScrollCorner(area, context, cornerRect);
}

void RenderScrollbarTheme::paintScrollbarBackground(GraphicsContext& context, Scrollbar& scrollbar)
{
    downcast<RenderScrollbar>(scrollbar).paintPart(context, ScrollbarBGPart, scrollbar.frameRect());
}

void RenderScrollbarTheme::paintTrackBackground(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& rect)
{
    downcast<RenderScrollbar>(scrollbar).paintPart(context, TrackBGPart, rect);
}

void RenderScrollbarTheme::paintTrackPiece(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& rect, ScrollbarPart part)
{
    downcast<RenderScrollbar>(scrollbar).paintPart(context, part, rect);
}

void RenderScrollbarTheme::paintButton(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& rect, ScrollbarPart part)
{
    downcast<RenderScrollbar>(scrollbar).paintPart(context, part, rect);
}

void RenderScrollbarTheme::paintThumb(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& rect)
{
    downcast<RenderScrollbar>(scrollbar).paintPart(context, ThumbPart, rect);
}

void RenderScrollbarTheme::paintTickmarks(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& rect)
{
    ScrollbarTheme::theme().paintTickmarks(context, scrollbar, rect);
}

}
