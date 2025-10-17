/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

#include "FloatRect.h"
#include "RenderReplaced.h"
#include "SVGBoundingBoxComputation.h"

namespace WebCore {

class RenderSVGViewportContainer;
class SVGSVGElement;

class RenderSVGRoot final : public RenderReplaced {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGRoot);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGRoot);
public:
    RenderSVGRoot(SVGSVGElement&, RenderStyle&&);
    virtual ~RenderSVGRoot();

    SVGSVGElement& svgSVGElement() const;
    FloatSize computeViewportSize() const;

    bool isEmbeddedThroughSVGImage() const;
    bool isEmbeddedThroughFrameContainingSVGDocument() const;

    void computeIntrinsicRatioInformation(FloatSize& intrinsicSize, FloatSize& intrinsicRatio) const final;
    bool hasIntrinsicAspectRatio() const final;

    bool isLayoutSizeChanged() const { return m_isLayoutSizeChanged; }
    bool didTransformToRootUpdate() const { return m_didTransformToRootUpdate; }
    bool isInLayout() const { return m_inLayout; }

    IntSize containerSize() const { return m_containerSize; }
    void setContainerSize(const IntSize& containerSize) { m_containerSize = containerSize; }

    bool hasRelativeDimensions() const final;

    bool shouldApplyViewportClip() const;

    FloatRect objectBoundingBox() const final { return m_objectBoundingBox; }
    FloatRect objectBoundingBoxWithoutTransformations() const final { return m_objectBoundingBoxWithoutTransformations; }
    FloatRect strokeBoundingBox() const final;
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final { return SVGBoundingBoxComputation::computeRepaintBoundingBox(*this); }

    LayoutRect visualOverflowRectEquivalent() const { return SVGBoundingBoxComputation::computeVisualOverflowRect(*this); }

    RenderSVGViewportContainer* viewportContainer() const;
    CheckedPtr<RenderSVGViewportContainer> checkedViewportContainer() const;

private:
    void element() const = delete;

    ASCIILiteral renderName() const final { return "RenderSVGRoot"_s; }
    bool requiresLayer() const final { return true; }

    bool updateLayoutSizeIfNeeded();
    bool paintingAffectedByExternalOffset() const;

    // To prevent certain legacy code paths to hit assertions in debug builds, when switching off LBSE (during the teardown of the LBSE tree).
    std::optional<FloatRect> computeFloatVisibleRectInContainer(const FloatRect&, const RenderLayerModelObject*, VisibleRectContext) const final { return std::nullopt; }

    LayoutUnit computeReplacedLogicalWidth(ShouldComputePreferred  = ShouldComputePreferred::ComputeActual) const final;
    LayoutUnit computeReplacedLogicalHeight(std::optional<LayoutUnit> estimatedUsedWidth = std::nullopt) const final;
    void layout() final;
    void layoutChildren();
    void paint(PaintInfo&, const LayoutPoint&) final;
    void paintObject(PaintInfo&, const LayoutPoint&) final;
    void paintContents(PaintInfo&, const LayoutPoint&);

    void willBeDestroyed() final;

    void updateFromStyle() final;
    bool needsHasSVGTransformFlags() const final;
    void updateLayerTransform() final;

    FloatSize calculateIntrinsicSize() const;

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) final;

    LayoutRect overflowClipRect(const LayoutPoint& location, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize, PaintPhase = PaintPhase::BlockBackground) const final;

    void mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState&, OptionSet<MapCoordinatesMode>, bool* wasFixed) const final;

    void boundingRects(Vector<LayoutRect>&, const LayoutPoint& accumulatedOffset) const final;
    void absoluteQuads(Vector<FloatQuad>&, bool* wasFixed) const final;

    bool canBeSelectionLeaf() const final { return false; }
    bool canHaveChildren() const final { return true; }

    bool m_inLayout { false };
    bool m_didTransformToRootUpdate { false };
    bool m_isLayoutSizeChanged { false };

    IntSize m_containerSize;
    FloatRect m_objectBoundingBox;
    FloatRect m_objectBoundingBoxWithoutTransformations;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_strokeBoundingBox;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGRoot, isRenderSVGRoot())
