/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
#include "SVGRenderSupport.h"
#include <wtf/WeakHashSet.h>

namespace WebCore {

class AffineTransform;
class LegacyRenderSVGResourceContainer;
class SVGSVGElement;

class LegacyRenderSVGRoot final : public RenderReplaced {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGRoot);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGRoot);
public:
    LegacyRenderSVGRoot(SVGSVGElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGRoot();

    SVGSVGElement& svgSVGElement() const;
    Ref<SVGSVGElement> protectedSVGSVGElement() const;

    bool isEmbeddedThroughSVGImage() const;
    bool isEmbeddedThroughFrameContainingSVGDocument() const;

    void computeIntrinsicRatioInformation(FloatSize& intrinsicSize, FloatSize& intrinsicRatio) const override;
    bool hasIntrinsicAspectRatio() const final;

    bool isLayoutSizeChanged() const { return m_isLayoutSizeChanged; }
    bool isInLayout() const { return m_inLayout; }
    void setNeedsBoundariesUpdate() override { m_needsBoundariesOrTransformUpdate = true; }
    void setNeedsTransformUpdate() override { m_needsBoundariesOrTransformUpdate = true; }

    IntSize containerSize() const { return m_containerSize; }
    void setContainerSize(const IntSize& containerSize) { m_containerSize = containerSize; }

    bool hasRelativeDimensions() const override;

    // localToBorderBoxTransform maps local SVG viewport coordinates to local CSS box coordinates.  
    const AffineTransform& localToBorderBoxTransform() const { return m_localToBorderBoxTransform; }

    // The flag is cleared at the beginning of each layout() pass. Elements then call this
    // method during layout when they are invalidated by a filter.
    static void addResourceForClientInvalidation(LegacyRenderSVGResourceContainer*);

private:
    void element() const = delete;

    // Intentially left 'RenderSVGRoot' instead of 'LegacyRenderSVGRoot', to avoid breaking layout tests.
    ASCIILiteral renderName() const override { return "RenderSVGRoot"_s; }

    LayoutUnit computeReplacedLogicalWidth(ShouldComputePreferred  = ShouldComputePreferred::ComputeActual) const override;
    LayoutUnit computeReplacedLogicalHeight(std::optional<LayoutUnit> estimatedUsedWidth = std::nullopt) const override;
    void layout() override;
    void paintReplaced(PaintInfo&, const LayoutPoint&) override;

    void willBeDestroyed() override;

    void insertedIntoTree() override;
    void willBeRemovedFromTree() override;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    const AffineTransform& localToParentTransform() const override;

    FloatRect objectBoundingBox() const override { return m_objectBoundingBox; }
    FloatRect strokeBoundingBox() const override;
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const override;

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;

    LayoutRect clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext) const override;
    RepaintRects rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds) const override;

    LayoutRect localClippedOverflowRect(RepaintRectCalculation) const;

    std::optional<FloatRect> computeFloatVisibleRectInContainer(const FloatRect&, const RenderLayerModelObject* container, VisibleRectContext) const override;

    void mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState&, OptionSet<MapCoordinatesMode>, bool* wasFixed) const override;
    const RenderObject* pushMappingToContainer(const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap&) const override;

    bool canBeSelectionLeaf() const override { return false; }
    bool canHaveChildren() const override { return true; }

    bool shouldApplyViewportClip() const;
    void updateCachedBoundaries();
    void buildLocalToBorderBoxTransform();

    FloatSize calculateIntrinsicSize() const;

    IntSize m_containerSize;
    FloatRect m_objectBoundingBox;
    bool m_objectBoundingBoxValid { false };
    bool m_inLayout { false };
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_strokeBoundingBox;
    FloatRect m_repaintBoundingBox;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_accurateRepaintBoundingBox;
    mutable AffineTransform m_localToParentTransform;
    AffineTransform m_localToBorderBoxTransform;
    SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer> m_resourcesNeedingToInvalidateClients;
    bool m_isLayoutSizeChanged : 1;
    bool m_needsBoundariesOrTransformUpdate : 1;
    bool m_hasBoxDecorations : 1;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGRoot, isLegacyRenderSVGRoot())
