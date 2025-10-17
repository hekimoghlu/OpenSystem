/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#include "RenderInline.h"

namespace WebCore {

class SVGGraphicsElement;

class RenderSVGInline : public RenderInline {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGInline);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGInline);
public:
    RenderSVGInline(Type, SVGGraphicsElement&, RenderStyle&&);
    virtual ~RenderSVGInline();

    inline SVGGraphicsElement& graphicsElement() const;

    bool isChildAllowed(const RenderObject&, const RenderStyle&) const override;

private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGInline"_s; }
    bool requiresLayer() const final { return false; }

    void updateFromStyle() final;

    // Chapter 10.4 of the SVG Specification say that we should use the
    // object bounding box of the parent text element.
    // We search for the root text element and take its bounding box.
    // It is also necessary to take the stroke and repaint rect of
    // this element, since we need it for filters.
    FloatRect objectBoundingBox() const final;
    FloatRect strokeBoundingBox() const final;
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final;

    LayoutPoint currentSVGLayoutLocation() const final { return { }; }
    void setCurrentSVGLayoutLocation(const LayoutPoint&) final { ASSERT_NOT_REACHED(); }

    bool needsHasSVGTransformFlags() const final;

    LayoutRect clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext) const final;
    RepaintRects rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds) const final;

    std::optional<FloatRect> computeFloatVisibleRectInContainer(const FloatRect&, const RenderLayerModelObject* container, VisibleRectContext) const final;

    void mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState&, OptionSet<MapCoordinatesMode>, bool* wasFixed) const final;
    const RenderObject* pushMappingToContainer(const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap&) const final;
    void absoluteQuads(Vector<FloatQuad>&, bool* wasFixed) const final;

    std::unique_ptr<LegacyInlineFlowBox> createInlineFlowBox() final;

    void willBeDestroyed() final;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGInline, isRenderSVGInline())
