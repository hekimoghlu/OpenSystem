/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#include "RenderBox.h"
#include "RenderLayerModelObject.h"
#include "SVGBoundingBoxComputation.h"
#include "SVGRenderSupport.h"

namespace WebCore {

// Most renderers in the SVG rendering tree will inherit from this class
// but not all. LegacyRenderSVGForeignObject, RenderSVGBlock, etc. inherit from
// existing RenderBlock classes, that all inherit from RenderLayerModelObject
// directly, without RenderSVGModelObject inbetween. Therefore code which
// needs to be shared between all SVG renderers goes to RenderLayerModelObject.
class SVGElement;

class RenderSVGModelObject : public RenderLayerModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGModelObject);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGModelObject);
public:
    virtual ~RenderSVGModelObject();

    bool requiresLayer() const override { return true; }

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    static bool checkIntersection(RenderElement*, const FloatRect&);
    static bool checkEnclosure(RenderElement*, const FloatRect&);

    inline SVGElement& element() const;

    LayoutRect currentSVGLayoutRect() const { return m_layoutRect; }
    void setCurrentSVGLayoutRect(const LayoutRect& layoutRect) { m_layoutRect = layoutRect; }

    LayoutPoint currentSVGLayoutLocation() const final { return m_layoutRect.location(); }
    void setCurrentSVGLayoutLocation(const LayoutPoint& location) final { m_layoutRect.setLocation(location); }

    // Mimic the RenderBox accessors - by sharing the same terminology the painting / hit testing / layout logic is
    // similar to read compared to non-SVG renderers such as RenderBox & friends.
    LayoutRect borderBoxRectEquivalent() const { return { LayoutPoint(), m_layoutRect.size() }; }
    LayoutRect contentBoxRectEquivalent() const { return borderBoxRectEquivalent(); }
    LayoutRect frameRectEquivalent() const { return m_layoutRect; }
    LayoutRect visualOverflowRectEquivalent() const { return SVGBoundingBoxComputation::computeVisualOverflowRect(*this); }
    LayoutSize locationOffsetEquivalent() const { return toLayoutSize(currentSVGLayoutLocation()); }

    bool hasVisualOverflow() const { return !borderBoxRectEquivalent().contains(visualOverflowRectEquivalent()); }

    // For RenderLayer only
    LayoutPoint topLeftLocationEquivalent() const { return currentSVGLayoutLocation(); }
    LayoutRect borderBoxRectInFragmentEquivalent(RenderFragmentContainer*, RenderBox::RenderBoxFragmentInfoFlags = RenderBox::RenderBoxFragmentInfoFlags::CacheRenderBoxFragmentInfo) const { return borderBoxRectEquivalent(); }
    virtual LayoutRect overflowClipRect(const LayoutPoint& location, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize, PaintPhase = PaintPhase::BlockBackground) const;
    LayoutRect overflowClipRectForChildLayers(const LayoutPoint& location, OverlayScrollbarSizeRelevancy relevancy) { return overflowClipRect(location, relevancy); }

    virtual Path computeClipPath(AffineTransform&) const;

protected:
    RenderSVGModelObject(Type, Document&, RenderStyle&&, OptionSet<SVGModelObjectFlag> = { });
    RenderSVGModelObject(Type, SVGElement&, RenderStyle&&, OptionSet<SVGModelObjectFlag> = { });

    void updateFromStyle() override;

    RepaintRects localRectsForRepaint(RepaintOutlineBounds) const override;
    std::optional<RepaintRects> computeVisibleRectsInContainer(const RepaintRects&, const RenderLayerModelObject* container, VisibleRectContext) const override;
    void mapAbsoluteToLocalPoint(OptionSet<MapCoordinatesMode>, TransformState&) const override;
    void mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState&, OptionSet<MapCoordinatesMode>, bool* wasFixed) const final;
    LayoutRect outlineBoundsForRepaint(const RenderLayerModelObject* repaintContainer, const RenderGeometryMap* = nullptr) const final;
    const RenderObject* pushMappingToContainer(const RenderLayerModelObject*, RenderGeometryMap&) const override;
    LayoutSize offsetFromContainer(RenderElement&, const LayoutPoint&, bool* offsetDependsOnPoint = nullptr) const override;

    void boundingRects(Vector<LayoutRect>&, const LayoutPoint& accumulatedOffset) const override;
    void absoluteQuads(Vector<FloatQuad>&, bool* wasFixed) const override;

    void addFocusRingRects(Vector<LayoutRect>&, const LayoutPoint& additionalOffset, const RenderLayerModelObject* paintContainer = 0) const override;
    void paintSVGOutline(PaintInfo&, const LayoutPoint& adjustedPaintOffset);

    // Returns false if the rect has no intersection with the applied clip rect. When the context specifies edge-inclusive
    // intersection, this return value allows distinguishing between no intersection and zero-area intersection.
    bool applyCachedClipAndScrollPosition(RepaintRects&, const RenderLayerModelObject* container, VisibleRectContext) const final;

private:
    LayoutSize cachedSizeForOverflowClip() const;

    LayoutRect m_layoutRect;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGModelObject, isRenderSVGModelObject())
