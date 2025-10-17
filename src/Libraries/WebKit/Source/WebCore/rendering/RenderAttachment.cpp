/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include "config.h"
#include "RenderAttachment.h"

#if ENABLE(ATTACHMENT_ELEMENT)

#include "FloatRect.h"
#include "FloatRoundedRect.h"
#include "FrameSelection.h"
#include "HTMLAttachmentElement.h"
#include "RenderBoxInlines.h"
#include "RenderChildIterator.h"
#include "RenderStyleSetters.h"
#include "RenderTheme.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderAttachment);

RenderAttachment::RenderAttachment(HTMLAttachmentElement& element, RenderStyle&& style)
    : RenderReplaced(Type::Attachment, element, WTFMove(style), LayoutSize())
    , m_isWideLayout(element.isWideLayout())
{
    ASSERT(isRenderAttachment());
#if ENABLE(SERVICE_CONTROLS)
    m_hasShadowControls = element.isImageMenuEnabled();
#endif
}

RenderAttachment::~RenderAttachment() = default;

HTMLAttachmentElement& RenderAttachment::attachmentElement() const
{
    return downcast<HTMLAttachmentElement>(nodeForNonAnonymous());
}

LayoutSize RenderAttachment::layoutWideLayoutAttachmentOnly()
{
    if (auto* wideLayoutShadowElement = attachmentElement().wideLayoutShadowContainer()) {
        if (auto* wideLayoutShadowRenderer = downcast<RenderBox>(wideLayoutShadowElement->renderer())) {
            if (wideLayoutShadowRenderer->needsLayout())
                wideLayoutShadowRenderer->layout();
            ASSERT(!wideLayoutShadowRenderer->needsLayout());
            return wideLayoutShadowRenderer->size();
        }
    }
    return { };
}

void RenderAttachment::layout()
{
    if (auto size = layoutWideLayoutAttachmentOnly(); !size.isEmpty()) {
        setIntrinsicSize(size);
        RenderReplaced::layout();
        if (hasShadowContent())
            layoutShadowContent(intrinsicSize());
        return;
    }

    LayoutSize newIntrinsicSize = theme().attachmentIntrinsicSize(*this);

    if (!theme().attachmentShouldAllowWidthToShrink(*this)) {
        m_minimumIntrinsicWidth = std::max(m_minimumIntrinsicWidth, newIntrinsicSize.width());
        newIntrinsicSize.setWidth(m_minimumIntrinsicWidth);
    }

    setIntrinsicSize(newIntrinsicSize);

    RenderReplaced::layout();
    
    if (hasShadowContent())
        layoutShadowContent(newIntrinsicSize);
}

LayoutUnit RenderAttachment::baselinePosition(FontBaseline, bool, LineDirectionMode, LinePositionMode) const
{
    if (auto* baselineElement = attachmentElement().wideLayoutImageElement()) {
        if (auto* baselineElementRenderBox = baselineElement->renderBox()) {
            // This is the bottom of the image assuming it is vertically centered.
            return (height() + baselineElementRenderBox->height()) / 2;
        }
        // Fallback to the bottom of the attachment if there is no image.
        return height();
    }

    return theme().attachmentBaseline(*this);
}

bool RenderAttachment::shouldDrawBorder() const
{
    if (style().usedAppearance() == StyleAppearance::BorderlessAttachment)
        return false;
    return m_shouldDrawBorder;
}

void RenderAttachment::paintReplaced(PaintInfo& paintInfo, const LayoutPoint& offset)
{
    if (paintInfo.phase != PaintPhase::Selection || !hasVisibleBoxDecorations() || !style().hasUsedAppearance())
        return;

    auto paintRect = borderBoxRect();
    paintRect.moveBy(offset);

    theme().paint(*this, paintInfo, paintRect);
}

void RenderAttachment::layoutShadowContent(const LayoutSize& size)
{
    for (auto& renderBox : childrenOfType<RenderBox>(*this)) {
        renderBox.mutableStyle().setHeight(Length(size.height(), LengthType::Fixed));
        renderBox.mutableStyle().setWidth(Length(size.width(), LengthType::Fixed));
        renderBox.setNeedsLayout(MarkOnlyThis);
        renderBox.layout();
    }
}

bool RenderAttachment::paintWideLayoutAttachmentOnly(const PaintInfo& paintInfo, const LayoutPoint& offset) const
{
    if (paintInfo.phase != PaintPhase::Foreground && paintInfo.phase != PaintPhase::Selection)
        return false;

    if (auto* wideLayoutShadowElement = attachmentElement().wideLayoutShadowContainer()) {
        if (auto* wideLayoutShadowRenderer = wideLayoutShadowElement->renderer()) {
            auto shadowPaintInfo = paintInfo;
            for (PaintPhase phase : { PaintPhase::BlockBackground, PaintPhase::ChildBlockBackgrounds, PaintPhase::Float, PaintPhase::Foreground, PaintPhase::Outline }) {
                shadowPaintInfo.phase = phase;
                wideLayoutShadowRenderer->paint(shadowPaintInfo, offset);
            }
        }

        attachmentElement().requestWideLayoutIconIfNeeded();
        return true;
    }
    return false;
}

} // namespace WebCore

#endif
