/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

#include "RenderFlexibleBox.h"
#include "RenderTextFragment.h"
#include <memory>

namespace WebCore {

class HTMLFormControlElement;
class RenderTextFragment;

// RenderButtons are just like normal flexboxes except that they will generate an anonymous block child.
// For inputs, they will also generate an anonymous RenderText and keep its style and content up
// to date as the button changes.
class RenderButton final : public RenderFlexibleBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderButton);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderButton);
public:
    RenderButton(HTMLFormControlElement&, RenderStyle&&);
    virtual ~RenderButton();

    HTMLFormControlElement& formControlElement() const;

    bool canBeSelectionLeaf() const override;

    bool createsAnonymousWrapper() const override { return true; }

    void updateFromElement() override;

    bool canHaveGeneratedChildren() const override;
    bool hasControlClip() const override;
    LayoutRect controlClipRect(const LayoutPoint&) const override;

    void updateAnonymousChildStyle(RenderStyle&) const override;

    void setText(const String&);
    String text() const;

#if PLATFORM(IOS_FAMILY)
    void layout() override;
#endif

    RenderTextFragment* textRenderer() const { return m_buttonText.get(); }

    RenderBlock* innerRenderer() const { return m_inner.get(); }
    void setInnerRenderer(RenderBlock&);

    LayoutUnit baselinePosition(FontBaseline, bool firstLine, LineDirectionMode, LinePositionMode = PositionOnContainingLine) const override;

private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderButton"_s; }

    bool hasLineIfEmpty() const override;

    bool isFlexibleBoxImpl() const override { return true; }

    SingleThreadWeakPtr<RenderTextFragment> m_buttonText;
    SingleThreadWeakPtr<RenderBlock> m_inner;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderButton, isRenderButton())
