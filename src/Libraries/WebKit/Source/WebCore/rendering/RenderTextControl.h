/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "RenderFlexibleBox.h"

namespace WebCore {

class TextControlInnerTextElement;
class HTMLTextFormControlElement;

class RenderTextControl : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTextControl);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTextControl);
public:
    virtual ~RenderTextControl();

    WEBCORE_EXPORT HTMLTextFormControlElement& textFormControlElement() const;

#if PLATFORM(IOS_FAMILY)
    bool canScroll() const;

    // Returns the line height of the inner renderer.
    int innerLineHeight() const override;
#endif

protected:
    RenderTextControl(Type, HTMLTextFormControlElement&, RenderStyle&&);

    // This convenience function should not be made public because innerTextElement may outlive the render tree.
    RefPtr<TextControlInnerTextElement> innerTextElement() const;

    int scrollbarThickness() const;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void hitInnerTextElement(HitTestResult&, const LayoutPoint& pointInContainer, const LayoutPoint& accumulatedOffset);

    float scaleEmToUnits(int x) const;

    virtual float getAverageCharWidth();
    virtual LayoutUnit preferredContentLogicalWidth(float charWidth) const = 0;
    virtual LayoutUnit computeControlLogicalHeight(LayoutUnit lineHeight, LayoutUnit nonContentHeight) const = 0;

    LogicalExtentComputedValues computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const override;
    void layoutExcludedChildren(bool relayoutChildren) override;

private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderTextControl"_s; }
    void computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const override;
    void computePreferredLogicalWidths() override;
    bool canHaveGeneratedChildren() const override { return false; }
    
    void addFocusRingRects(Vector<LayoutRect>&, const LayoutPoint& additionalOffset, const RenderLayerModelObject* paintContainer = 0) const override;

    bool canBeProgramaticallyScrolled() const override { return true; }
};

// Renderer for our inner container, for <search> and others.
// We can't use RenderFlexibleBox directly, because flexboxes have a different
// baseline definition, and then inputs of different types wouldn't line up
// anymore.
class RenderTextControlInnerContainer final : public RenderFlexibleBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTextControlInnerContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTextControlInnerContainer);
public:
    RenderTextControlInnerContainer(Element&, RenderStyle&&);
    virtual ~RenderTextControlInnerContainer();

    LayoutUnit baselinePosition(FontBaseline baseline, bool firstLine, LineDirectionMode direction, LinePositionMode position) const override
    {
        return RenderBlock::baselinePosition(baseline, firstLine, direction, position);
    }
    std::optional<LayoutUnit> firstLineBaseline() const override { return RenderBlock::firstLineBaseline(); }
    std::optional<LayoutUnit> inlineBlockBaseline(LineDirectionMode direction) const override { return RenderBlock::inlineBlockBaseline(direction); }

private:
    bool isFlexibleBoxImpl() const override { return true; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTextControl, isRenderTextControl())
SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTextControlInnerContainer, isRenderTextControlInnerContainer())
