/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#include "Adwaita.h"

#if USE(THEME_ADWAITA)

#include "GraphicsContext.h"
#include "ThemeAdwaita.h"

namespace WebCore::Adwaita {

static inline float getRectRadius(const FloatRect& rect, int offset)
{
    return (std::min(rect.width(), rect.height()) + offset) / 2;
}

void paintFocus(GraphicsContext& graphicsContext, const FloatRect& rect, int offset, const Color& color, PaintRounded rounded)
{
    FloatRect focusRect = rect;
    focusRect.inflate(offset);

    float radius = (rounded == PaintRounded::Yes) ? getRectRadius(rect, offset) : 2;

    Path path;
    path.addRoundedRect(focusRect, { radius, radius });
    paintFocus(graphicsContext, path, color);
}

void paintFocus(GraphicsContext& graphicsContext, const Path& path, const Color& color)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    graphicsContext.beginTransparencyLayer(color.alphaAsFloat());
    // Since we cut off a half of it by erasing the rect contents, and half
    // of the stroke ends up inside that area, it needs to be twice as thick.
    graphicsContext.setStrokeThickness(focusLineWidth * 2);
    graphicsContext.setLineCap(LineCap::Round);
    graphicsContext.setLineJoin(LineJoin::Round);
    graphicsContext.setStrokeColor(color.opaqueColor());
    graphicsContext.strokePath(path);
    graphicsContext.setFillRule(WindRule::NonZero);
    graphicsContext.setCompositeOperation(CompositeOperator::Clear);
    graphicsContext.fillPath(path);
    graphicsContext.setCompositeOperation(CompositeOperator::SourceOver);
    graphicsContext.endTransparencyLayer();
}

void paintFocus(GraphicsContext& graphicsContext, const Vector<FloatRect>& rects, const Color& color, PaintRounded rounded)
{
    Path path;
    for (const auto& rect : rects) {
        float radius = (rounded == PaintRounded::Yes) ? getRectRadius(rect, 0) : 2;

        path.addRoundedRect(rect, { radius, radius });
    }
    paintFocus(graphicsContext, path, color);
}

void paintArrow(GraphicsContext& graphicsContext, const FloatRect& rect, ArrowDirection direction, bool useDarkAppearance)
{
    auto offset = rect.location();
    float size;
    if (rect.width() > rect.height()) {
        size = rect.height();
        offset.move((rect.width() - size) / 2, 0);
    } else {
        size = rect.width();
        offset.move(0, (rect.height() - size) / 2);
    }
    float zoomFactor = size / arrowSize;
    auto transform = [&](FloatPoint point) {
        point.scale(zoomFactor);
        point.moveBy(offset);
        return point;
    };
    Path path;
    switch (direction) {
    case ArrowDirection::Down:
        path.moveTo(transform({ 3, 6 }));
        path.addLineTo(transform({ 13, 6 }));
        path.addLineTo(transform({ 8, 11 }));
        break;
    case ArrowDirection::Up:
        path.moveTo(transform({ 3, 10 }));
        path.addLineTo(transform({ 8, 5 }));
        path.addLineTo(transform({ 13, 10 }));
        break;
    }
    path.closeSubpath();

    graphicsContext.setFillColor(useDarkAppearance ? arrowColorDark : arrowColorLight);
    graphicsContext.fillPath(path);
}

Color systemAccentColor()
{
    return static_cast<ThemeAdwaita&>(Theme::singleton()).accentColor();
}

Color systemFocusRingColor()
{
    return focusColor(systemAccentColor());
}

} // namespace WebCore::Adwaita

#endif // USE(THEME_ADWAITA)
