/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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

#include "RenderBlockFlow.h"

namespace WebCore {

class SVGElement;
class SVGGraphicsElement;

class RenderSVGBlock : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGBlock);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGBlock);
public:
    inline SVGGraphicsElement& graphicsElement() const;
    inline Ref<SVGGraphicsElement> protectedGraphicsElement() const;

protected:
    RenderSVGBlock(Type, SVGGraphicsElement&, RenderStyle&&);
    virtual ~RenderSVGBlock();

    void willBeDestroyed() override;

    void computeOverflow(LayoutUnit oldClientAfterEdge, bool recomputeFloats = false) override;

    void updateFromStyle() override;
    bool needsHasSVGTransformFlags() const override;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

private:
    void element() const = delete;

    void boundingRects(Vector<LayoutRect>&, const LayoutPoint& accumulatedOffset) const override;
    void absoluteQuads(Vector<FloatQuad>&, bool* wasFixed) const override;

    LayoutPoint currentSVGLayoutLocation() const final { return location(); }
    void setCurrentSVGLayoutLocation(const LayoutPoint& location) final { setLocation(location); }

    FloatRect referenceBoxRect(CSSBoxType) const final;

    LayoutRect clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext) const final;
    RepaintRects rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds) const final;

    std::optional<FloatRect> computeFloatVisibleRectInContainer(const FloatRect&, const RenderLayerModelObject* container, VisibleRectContext) const final;
    std::optional<RepaintRects> computeVisibleRectsInContainer(const RepaintRects&, const RenderLayerModelObject* container, VisibleRectContext) const final;

    void mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState&, OptionSet<MapCoordinatesMode>, bool* wasFixed) const final;
    const RenderObject* pushMappingToContainer(const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap&) const final;
    LayoutSize offsetFromContainer(RenderElement&, const LayoutPoint&, bool* offsetDependsOnPoint = nullptr) const override;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGBlock, isRenderSVGBlock())
