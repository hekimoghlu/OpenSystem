/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

#include "LayoutRepainter.h"
#include "PaintInfo.h"
#include "RenderObject.h"

namespace WebCore {

class FloatPoint;
class FloatRect;
class ImageBuffer;
class LayoutRect;
class RenderBoxModelObject;
class RenderElement;
class RenderGeometryMap;
class RenderLayerModelObject;
class RenderStyle;
class LegacyRenderSVGRoot;
class SVGElement;
class TransformState;

// SVGRendererSupport is a helper class sharing code between all SVG renderers.
class SVGRenderSupport {
public:
    static void layoutDifferentRootIfNeeded(const RenderElement&);

    // Shares child layouting code between LegacyRenderSVGRoot/RenderSVG(Hidden)Container
    static void layoutChildren(RenderElement&, bool selfNeedsLayout);

    // Helper function determining wheter overflow is hidden
    static bool isOverflowHidden(const RenderElement&);

    // Calculates the repaintRect in combination with filter, clipper and masker in local coordinates.
    static void intersectRepaintRectWithResources(const RenderElement&, FloatRect&, RepaintRectCalculation = RepaintRectCalculation::Fast);

    // Determines whether a container needs to be laid out because it's filtered and a child is being laid out.
    static bool filtersForceContainerLayout(const RenderElement&);

    // Determines whether the passed point lies in a clipping area
    static bool pointInClippingArea(const RenderElement&, const FloatPoint&);

    static void computeContainerBoundingBoxes(const RenderElement& container, FloatRect& objectBoundingBox, bool& objectBoundingBoxValid, FloatRect& repaintBoundingBox, RepaintRectCalculation = RepaintRectCalculation::Fast);
    static FloatRect computeContainerStrokeBoundingBox(const RenderElement& container);
    static bool paintInfoIntersectsRepaintRect(const FloatRect& localRepaintRect, const AffineTransform& localTransform, const PaintInfo&);

    // Important functions used by nearly all SVG renderers centralizing coordinate transformations / repaint rect calculations
    static LayoutRect clippedOverflowRectForRepaint(const RenderElement&, const RenderLayerModelObject* container, RenderObject::VisibleRectContext);
    static std::optional<FloatRect> computeFloatVisibleRectInContainer(const RenderElement&, const FloatRect&, const RenderLayerModelObject* container, RenderObject::VisibleRectContext);
    static const RenderElement& localToParentTransform(const RenderElement&, AffineTransform&);
    static void mapLocalToContainer(const RenderElement&, const RenderLayerModelObject* ancestorContainer, TransformState&, bool* wasFixed);
    static const RenderElement* pushMappingToContainer(const RenderElement&, const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap&);
    static LayoutRepainter::CheckForRepaint checkForSVGRepaintDuringLayout(const RenderElement&);

    static FloatRect calculateApproximateStrokeBoundingBox(const RenderElement&);

    // Shared between SVG renderers and resources.
    static void applyStrokeStyleToContext(GraphicsContext&, const RenderStyle&, const RenderElement&);

    // Determines if any ancestor's transform has changed.
    static bool transformToRootChanged(RenderElement*);

    static void clipContextToCSSClippingArea(GraphicsContext&, const RenderElement& renderer);

    static void styleChanged(RenderElement&, const RenderStyle*);

    static bool isolatesBlending(const RenderStyle&);
    static void updateMaskedAncestorShouldIsolateBlending(const RenderElement&);

    static LegacyRenderSVGRoot* findTreeRootObject(RenderElement&);
    static const LegacyRenderSVGRoot* findTreeRootObject(const RenderElement&);

private:
    // This class is not constructable.
    SVGRenderSupport();
    ~SVGRenderSupport();
};

} // namespace WebCore
