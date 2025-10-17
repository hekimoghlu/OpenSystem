/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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

#if ENABLE(ATTACHMENT_ELEMENT)

#include "HTMLAttachmentElement.h"
#include "RenderReplaced.h"

namespace WebCore {

class RenderAttachment final : public RenderReplaced {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderAttachment);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderAttachment);
public:
    RenderAttachment(HTMLAttachmentElement&, RenderStyle&&);
    virtual ~RenderAttachment();

    HTMLAttachmentElement& attachmentElement() const;

    void setShouldDrawBorder(bool drawBorder) { m_shouldDrawBorder = drawBorder; }
    bool shouldDrawBorder() const;

    void setHasShadowControls(bool hasShadowControls) { m_hasShadowControls = hasShadowControls; }
    bool hasShadowControls() const { return m_hasShadowControls; }
    bool isWideLayout() const { return m_isWideLayout; }
    bool hasShadowContent() const { return hasShadowControls() || isWideLayout(); }
    bool canHaveGeneratedChildren() const override { return hasShadowContent(); }
    bool canHaveChildren() const override { return hasShadowContent(); }

    bool paintWideLayoutAttachmentOnly(const PaintInfo&, const LayoutPoint& offset) const;

private:
    void element() const = delete;
    ASCIILiteral renderName() const override { return "RenderAttachment"_s; }
    LayoutSize layoutWideLayoutAttachmentOnly();
    void layoutShadowContent(const LayoutSize&) override;

    bool shouldDrawSelectionTint() const override { return isWideLayout(); }
    void paintReplaced(PaintInfo&, const LayoutPoint& offset) final;

    void layout() override;

    LayoutUnit baselinePosition(FontBaseline, bool, LineDirectionMode, LinePositionMode) const override;

    LayoutUnit m_minimumIntrinsicWidth;
    bool m_shouldDrawBorder { true };
    bool m_hasShadowControls { false };
    bool m_isWideLayout;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderAttachment, isRenderAttachment())

#endif // ENABLE(ATTACHMENT_ELEMENT)
