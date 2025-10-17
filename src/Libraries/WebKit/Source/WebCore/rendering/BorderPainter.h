/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#include "RenderBoxModelObject.h"
#include "RenderElement.h"

namespace WebCore {

class BorderPainter {
public:
    BorderPainter(const RenderElement&, const PaintInfo&);

    void paintBorder(const LayoutRect&, const RenderStyle&, BleedAvoidance = BleedAvoidance::None, RectEdges<bool> closedEdges = { true }) const;
    void paintOutline(const LayoutRect&) const;
    void paintOutline(const LayoutPoint& paintOffset, const Vector<LayoutRect>& lineRects) const;

    bool paintNinePieceImage(const LayoutRect&, const RenderStyle&, const NinePieceImage&, CompositeOperator = CompositeOperator::SourceOver) const;
    static void drawLineForBoxSide(GraphicsContext&, const Document&, const FloatRect&, BoxSide, Color, BorderStyle, float adjacentWidth1, float adjacentWidth2, bool antialias = false);

    static std::optional<Path> pathForBorderArea(const LayoutRect&, const RenderStyle&, float deviceScaleFactor, RectEdges<bool> closedEdges = { true });

    static bool allCornersClippedOut(const RoundedRect&, const LayoutRect&);
    static bool shouldAntialiasLines(GraphicsContext&);

private:
    struct Sides;
    void paintSides(const Sides&) const;

    void paintTranslucentBorderSides(const RoundedRect& outerBorder, const RoundedRect& innerBorder, const IntPoint& innerBorderAdjustment,
        const BorderEdges&, BoxSideSet edgesToDraw, std::optional<BorderDataRadii>, BleedAvoidance, RectEdges<bool> closedEdges, bool antialias) const;
    void paintBorderSides(const RoundedRect& outerBorder, const RoundedRect& innerBorder, const IntPoint& innerBorderAdjustment, const BorderEdges&, BoxSideSet edgeSet, std::optional<BorderDataRadii>, BleedAvoidance, RectEdges<bool> closedEdges, bool antialias, const Color* overrideColor = nullptr) const;
    void paintOneBorderSide(const RoundedRect& outerBorder, const RoundedRect& innerBorder,
        const LayoutRect& sideRect, BoxSide, BoxSide adjacentSide1, BoxSide adjacentSide2, const BorderEdges&, std::optional<BorderDataRadii>, const Path*, BleedAvoidance, RectEdges<bool> closedEdges, bool antialias, const Color* overrideColor) const;
    void drawBoxSideFromPath(const LayoutRect& borderRect, const Path& borderPath, const BorderEdges&, std::optional<BorderDataRadii>, float thickness, float drawThickness, BoxSide, Color, BorderStyle, BleedAvoidance, RectEdges<bool> closedEdges) const;
    void clipBorderSidePolygon(const RoundedRect& outerBorder, const RoundedRect& innerBorder, BoxSide, bool firstEdgeMatches, bool secondEdgeMatches) const;

    LayoutRect borderRectAdjustedForBleedAvoidance(const LayoutRect&, BleedAvoidance) const;

    static Color calculateBorderStyleColor(const BorderStyle&, const BoxSide&, const Color&);

    const Document& document() const;

    CheckedRef<const RenderElement> m_renderer;
    const PaintInfo& m_paintInfo;
};

LayoutRect shrinkRectByOneDevicePixel(const GraphicsContext&, const LayoutRect&, float devicePixelRatio);

}
