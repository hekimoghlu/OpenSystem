/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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

#include "HTMLInputElement.h"
#include "RenderTextControl.h"

namespace WebCore {

class RenderTextControlSingleLine : public RenderTextControl {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTextControlSingleLine);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTextControlSingleLine);
public:
    RenderTextControlSingleLine(Type, HTMLInputElement&, RenderStyle&&);
    virtual ~RenderTextControlSingleLine();

protected:
    HTMLElement* containerElement() const;
    HTMLElement* innerBlockElement() const;
    HTMLInputElement& inputElement() const;
    Ref<HTMLInputElement> protectedInputElement() const;

private:
    void textFormControlElement() const = delete;

    bool hasControlClip() const override;
    LayoutRect controlClipRect(const LayoutPoint&) const override;

    void layout() override;

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;

    void autoscroll(const IntPoint&) override;

    // Subclassed to forward to our inner div.
    int scrollLeft() const override;
    int scrollTop() const override;
    int scrollWidth() const override;
    int scrollHeight() const override;
    void setScrollLeft(int, const ScrollPositionChangeOptions&) override;
    void setScrollTop(int, const ScrollPositionChangeOptions&) override;
    bool scroll(ScrollDirection, ScrollGranularity, unsigned stepCount = 1, Element** stopElement = nullptr, RenderBox* startBox = nullptr, const IntPoint& wheelEventAbsolutePoint = IntPoint()) final;
    bool logicalScroll(ScrollLogicalDirection, ScrollGranularity, unsigned stepCount = 1, Element** stopElement = nullptr) final;

    int textBlockWidth() const;
    float getAverageCharWidth() override;
    LayoutUnit preferredContentLogicalWidth(float charWidth) const override;
    LayoutUnit computeControlLogicalHeight(LayoutUnit lineHeight, LayoutUnit nonContentHeight) const override;
    
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    HTMLElement* innerSpinButtonElement() const;
};

inline HTMLElement* RenderTextControlSingleLine::containerElement() const
{
    return inputElement().containerElement();
}

inline HTMLElement* RenderTextControlSingleLine::innerBlockElement() const
{
    return inputElement().innerBlockElement();
}

// ----------------------------

class RenderTextControlInnerBlock final : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTextControlInnerBlock);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTextControlInnerBlock);
public:
    RenderTextControlInnerBlock(Element&, RenderStyle&&);
    virtual ~RenderTextControlInnerBlock();

private:
    bool hasLineIfEmpty() const override { return true; }
    bool canBeProgramaticallyScrolled() const override
    {
        if (auto* shadowHost = dynamicDowncast<HTMLInputElement>(element()->shadowHost()))
            return !shadowHost->hasAutofillStrongPasswordButton();
        return true;
    }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTextControlSingleLine, isRenderTextControlSingleLine())
SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTextControlInnerBlock, isRenderTextControlInnerBlock())
