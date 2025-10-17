/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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

#include "AffineTransform.h"
#include "RenderSVGBlock.h"
#include "SVGBoundingBoxComputation.h"
#include "SVGTextChunk.h"
#include "SVGTextLayoutAttributesBuilder.h"

namespace WebCore {

namespace InlineIterator {
class InlineBoxIterator;
}

class RenderSVGInlineText;
class SVGRootInlineBox;
class SVGTextElement;
class SVGTextLayoutEngine;

class RenderSVGText final : public RenderSVGBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGText);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGText);
public:
    RenderSVGText(SVGTextElement&, RenderStyle&&);
    virtual ~RenderSVGText();

    SVGTextElement& textElement() const;
    Ref<SVGTextElement> protectedTextElement() const;

    bool isChildAllowed(const RenderObject&, const RenderStyle&) const override;

    void setNeedsPositioningValuesUpdate() { m_needsPositioningValuesUpdate = true; }
    void setNeedsTextMetricsUpdate() { m_needsTextMetricsUpdate = true; }

    // FIXME: [LBSE] Only needed for legacy SVG engine.
    void setNeedsTransformUpdate() override { m_needsTransformUpdate = true; }

    static RenderSVGText* locateRenderSVGTextAncestor(RenderObject&);
    static const RenderSVGText* locateRenderSVGTextAncestor(const RenderObject&);

    bool needsReordering() const { return m_needsReordering; }
    Vector<SVGTextLayoutAttributes*>& layoutAttributes() { return m_layoutAttributes; }

    void subtreeChildWasAdded(RenderObject*);
    void subtreeChildWillBeRemoved(RenderObject*, Vector<SVGTextLayoutAttributes*, 2>& affectedAttributes);
    void subtreeChildWasRemoved(const Vector<SVGTextLayoutAttributes*, 2>& affectedAttributes);
    void subtreeTextDidChange(RenderSVGInlineText*);

    FloatRect objectBoundingBox() const final { return m_objectBoundingBox; }
    FloatRect strokeBoundingBox() const final;
    bool isObjectBoundingBoxValid() const;
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final;

    LayoutRect visualOverflowRectEquivalent() const { return SVGBoundingBoxComputation::computeVisualOverflowRect(*this); }

    void updatePositionAndOverflow(const FloatRect&);

    SVGRootInlineBox* legacyRootBox() const;

private:
    void graphicsElement() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGText"_s; }

    void paint(PaintInfo&, const LayoutPoint&) override;
    void paintInlineChildren(PaintInfo&, const LayoutPoint&) override;

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;
    bool hitTestInlineChildren(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;

    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const final;
    VisiblePosition positionForPoint(const LayoutPoint&, HitTestSource, const RenderFragmentContainer*) override;

    bool requiresLayer() const override
    {
        if (document().settings().layerBasedSVGEngineEnabled())
            return true;
        return false;
    }

    void layout() override;

    void computePerCharacterLayoutInformation();
    void layoutCharactersInTextBoxes(const InlineIterator::InlineBoxIterator&, SVGTextLayoutEngine&);
    FloatRect layoutChildBoxes(LegacyInlineFlowBox*, SVGTextFragmentMap&);
    void layoutRootBox(const FloatRect&);
    void reorderValueListsToLogicalOrder();

    void willBeDestroyed() override;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;

    // FIXME: [LBSE] Begin code only needed for legacy SVG engine.
    bool nodeAtFloatPoint(const HitTestRequest&, HitTestResult&, const FloatPoint& pointInParent, HitTestAction) override;
    const AffineTransform& localToParentTransform() const override { return m_localTransform; }
    AffineTransform localTransform() const override { return m_localTransform; }
    // FIXME: [LBSE] End code only needed for legacy SVG engine.

    bool shouldHandleSubtreeMutations() const;

    bool m_needsReordering : 1 { false };
    bool m_needsPositioningValuesUpdate : 1 { false };
    bool m_needsTransformUpdate : 1 { true }; // FIXME: [LBSE] Only needed for legacy SVG engine.
    bool m_needsTextMetricsUpdate : 1 { false };
    bool m_hasPerformedLayout : 1 { false }; // Needed to distinguish between when we perform a full pass of layout and everHadLayout (which can be set be content visibility for skipped content).
    AffineTransform m_localTransform; // FIXME: [LBSE] Only needed for legacy SVG engine.
    SVGTextLayoutAttributesBuilder m_layoutAttributesBuilder;
    Vector<SVGTextLayoutAttributes*> m_layoutAttributes;
    FloatRect m_objectBoundingBox;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGText, isRenderSVGText())
