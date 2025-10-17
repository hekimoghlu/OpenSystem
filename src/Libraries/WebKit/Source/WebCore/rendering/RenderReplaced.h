/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

namespace WebCore {

class RenderReplaced : public RenderBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderReplaced);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderReplaced);
public:
    virtual ~RenderReplaced();

    LayoutUnit computeReplacedLogicalWidth(ShouldComputePreferred = ShouldComputePreferred::ComputeActual) const override;
    LayoutUnit computeReplacedLogicalHeight(std::optional<LayoutUnit> estimatedUsedWidth = std::nullopt) const override;

    LayoutRect replacedContentRect(const LayoutSize& intrinsicSize) const;
    LayoutRect replacedContentRect() const { return replacedContentRect(intrinsicSize()); }

    bool setNeedsLayoutIfNeededAfterIntrinsicSizeChange();

    LayoutSize intrinsicSize() const final;
    FloatSize intrinsicRatio() const;
    
    bool isContentLikelyVisibleInViewport();
    bool needsPreferredWidthsRecalculation() const override;

    double computeIntrinsicAspectRatio() const;

    void computeIntrinsicRatioInformation(FloatSize& intrinsicSize, FloatSize& intrinsicRatio) const override;

    virtual bool paintsContent() const { return true; }

protected:
    RenderReplaced(Type, Element&, RenderStyle&&, OptionSet<ReplacedFlag> = { });
    RenderReplaced(Type, Element&, RenderStyle&&, const LayoutSize& intrinsicSize, OptionSet<ReplacedFlag> = { });
    RenderReplaced(Type, Document&, RenderStyle&&, const LayoutSize& intrinsicSize, OptionSet<ReplacedFlag> = { });

    void layout() override;

    void computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const final;

    virtual LayoutUnit minimumReplacedHeight() const { return 0_lu; }

    bool isSelected() const;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void setIntrinsicSize(const LayoutSize& intrinsicSize) { m_intrinsicSize = intrinsicSize; }
    virtual void intrinsicSizeChanged();
    virtual bool hasRelativeIntrinsicLogicalWidth() const { return false; }

    void paint(PaintInfo&, const LayoutPoint&) override;
    bool shouldPaint(PaintInfo&, const LayoutPoint&);
    LayoutRect localSelectionRect(bool checkWhetherSelected = true) const; // This is in local coordinates, but it's a physical rect (so the top left corner is physical top left).

    void willBeDestroyed() override;

    virtual void layoutShadowContent(const LayoutSize&);

private:
    LayoutUnit computeConstrainedLogicalWidth() const;

    virtual RenderBox* embeddedContentBox() const { return 0; }
    ASCIILiteral renderName() const override { return "RenderReplaced"_s; }

    bool canHaveChildren() const override { return false; }

    void computePreferredLogicalWidths() final;
    virtual void paintReplaced(PaintInfo&, const LayoutPoint&) { }

    RepaintRects localRectsForRepaint(RepaintOutlineBounds) const override;

    VisiblePosition positionForPoint(const LayoutPoint&, HitTestSource, const RenderFragmentContainer*) final;
    
    bool canBeSelectionLeaf() const override { return true; }

    LayoutRect selectionRectForRepaint(const RenderLayerModelObject* repaintContainer, bool clipToVisibleContent = true) final;
    void computeAspectRatioInformationForRenderBox(RenderBox*, FloatSize& constrainedSize, FloatSize& intrinsicRatio) const;
    void computeIntrinsicSizesConstrainedByTransferredMinMaxSizes(RenderBox* contentRenderer, FloatSize& constrainedSize, FloatSize& intrinsicRatio) const;

    virtual bool shouldDrawSelectionTint() const;
    
    Color calculateHighlightColor() const;
    bool isHighlighted(HighlightState, const RenderHighlight&) const;

    bool hasReplacedLogicalHeight() const;

    mutable LayoutSize m_intrinsicSize;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderReplaced, isRenderReplaced())
