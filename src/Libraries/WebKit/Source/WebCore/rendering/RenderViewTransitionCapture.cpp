/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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
#include "RenderViewTransitionCapture.h"

#include "ImageOverlayController.h"
#include "ImageQualityController.h"
#include "PaintInfo.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderViewTransitionCapture);

RenderViewTransitionCapture::RenderViewTransitionCapture(Type type, Document& document, RenderStyle&& style)
    : RenderReplaced(type, document, WTFMove(style), { }, ReplacedFlag::IsViewTransitionCapture)
{
}

RenderViewTransitionCapture::~RenderViewTransitionCapture() = default;

void RenderViewTransitionCapture::setImage(RefPtr<ImageBuffer> oldImage)
{
    m_oldImage = oldImage;
    if (hasLayer())
        layer()->contentChanged(ContentChangeType::Image);
    if (parent())
        repaint();
}

bool RenderViewTransitionCapture::setCapturedSize(const LayoutSize& size, const LayoutRect& overflowRect, const LayoutPoint& layerToLayoutOffset)
{
    if (m_overflowRect == overflowRect && intrinsicSize() == size && m_layerToLayoutOffset == layerToLayoutOffset)
        return false;
    m_imageIntrinsicSize = size;
    setIntrinsicSize(size);
    m_overflowRect = overflowRect;
    m_layerToLayoutOffset = layerToLayoutOffset;
    return true;
}

void RenderViewTransitionCapture::intrinsicSizeChanged()
{
    if (intrinsicSize() == m_imageIntrinsicSize)
        return;
    setIntrinsicSize(m_imageIntrinsicSize);
    setPreferredLogicalWidthsDirty(true);
    setNeedsLayout();
}

void RenderViewTransitionCapture::paintReplaced(PaintInfo& paintInfo, const LayoutPoint& paintOffset)
{
    auto& context = paintInfo.context();
    if (context.detectingContentfulPaint())
        return;

    LayoutRect replacedContentRect = this->replacedContentRect();
    replacedContentRect.moveBy(paintOffset);

    FloatRect paintRect = m_localOverflowRect;

    InterpolationQualityMaintainer interpolationMaintainer(context, ImageQualityController::interpolationQualityFromStyle(style()));
    if (m_oldImage)
        context.drawImageBuffer(*m_oldImage, paintRect, { context.compositeOperation() });
}

void RenderViewTransitionCapture::layout()
{
    RenderReplaced::layout();
    // Move the overflow rect of the captured renderer into layout coords, and then scale/position so that the intrinsic size subset covers
    // our replaced content rect.
    m_localOverflowRect = m_overflowRect;
    m_localOverflowRect.moveBy(-m_layerToLayoutOffset);
    m_scale = { replacedContentRect().width().toFloat() / intrinsicSize().width().toFloat() , replacedContentRect().height().toFloat() / intrinsicSize().height().toFloat()  };
    m_localOverflowRect.scale(m_scale.width(), m_scale.height());
    m_localOverflowRect.moveBy(replacedContentRect().location());

    addVisualOverflow(m_localOverflowRect);
}

void RenderViewTransitionCapture::updateFromStyle()
{
    RenderReplaced::updateFromStyle();

    if (effectiveOverflowX() != Overflow::Visible || effectiveOverflowY() != Overflow::Visible)
        setHasNonVisibleOverflow();
}

LayoutPoint RenderViewTransitionCapture::captureContentInset() const
{
    LayoutPoint location = m_localOverflowRect.location();
    location.moveBy(-visualOverflowRect().location());
    return location;
}

Node* RenderViewTransitionCapture::nodeForHitTest() const
{
    // The view transition pseudo-elements should hit-test to their originating element (the document element).
    return document().documentElement();
}

bool RenderViewTransitionCapture::paintsContent() const
{
    if (style().pseudoElementType() == PseudoId::ViewTransitionOld)
        return true;
    return !canUseExistingLayers();
}

String RenderViewTransitionCapture::debugDescription() const
{
    StringBuilder builder;

    builder.append(renderName(), " 0x"_s, hex(reinterpret_cast<uintptr_t>(this), Lowercase));

    builder.append(" ::view-transition-"_s, style().pseudoElementType() == PseudoId::ViewTransitionNew ? "new("_s : "old("_s);
    builder.append(style().pseudoElementNameArgument(), ')');
    return builder.toString();
}

} // namespace WebCore
