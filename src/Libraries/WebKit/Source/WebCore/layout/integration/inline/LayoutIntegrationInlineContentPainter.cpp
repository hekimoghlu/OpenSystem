/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
#include "LayoutIntegrationInlineContentPainter.h"

#include "EllipsisBoxPainter.h"
#include "InlineBoxPainter.h"
#include "PaintInfo.h"
#include "RenderBox.h"
#include "RenderInline.h"
#include "RenderStyleInlines.h"
#include "TextBoxPainter.h"
#include <wtf/Assertions.h>

namespace WebCore {
namespace LayoutIntegration {

InlineContentPainter::InlineContentPainter(PaintInfo& paintInfo, const LayoutPoint& paintOffset, const RenderInline* inlineBoxWithLayer, const InlineContent& inlineContent, const RenderBlockFlow& root)
    : m_paintInfo(paintInfo)
    , m_paintOffset(paintOffset)
    , m_inlineBoxWithLayer(inlineBoxWithLayer)
    , m_inlineContent(inlineContent)
    , m_root(root)
{
    m_damageRect = m_paintInfo.rect;
    m_damageRect.moveBy(-m_paintOffset);
}

void InlineContentPainter::paintEllipsis(size_t lineIndex)
{
    if ((m_paintInfo.phase != PaintPhase::Foreground && m_paintInfo.phase != PaintPhase::TextClip) || root().style().usedVisibility() != Visibility::Visible)
        return;

    auto lineBox = InlineIterator::LineBox { InlineIterator::LineBoxIteratorModernPath { m_inlineContent, lineIndex } };
    if (!lineBox.hasEllipsis())
        return;
    EllipsisBoxPainter { lineBox, m_paintInfo, m_paintOffset, root().selectionForegroundColor(), root().selectionBackgroundColor() }.paint();
}

void InlineContentPainter::paintDisplayBox(const InlineDisplay::Box& box)
{
    auto hasDamage = [&](auto& box) {
        auto rect = enclosingLayoutRect(box.inkOverflow());
        root().flipForWritingMode(rect);
        // FIXME: This should test for intersection but horizontal ink overflow is miscomputed in a few cases (like with negative letter-spacing).
        return m_damageRect.maxY() > rect.y() && m_damageRect.y() < rect.maxY();
    };

    if (box.isFullyTruncated()) {
        // Fully truncated boxes are visually empty and they don't show their descendants either (unlike visibility property). 
        return;
    }

    if (box.isLineBreak())
        return;

    if (box.isInlineBox()) {
        if (!box.isVisible() || !hasDamage(box))
            return;

        auto inlineBoxPaintInfo = PaintInfo { m_paintInfo };
        inlineBoxPaintInfo.phase = m_paintInfo.phase == PaintPhase::ChildOutlines ? PaintPhase::Outline : m_paintInfo.phase;
        inlineBoxPaintInfo.outlineObjects = &m_outlineObjects;

        InlineBoxPainter { m_inlineContent, box, inlineBoxPaintInfo, m_paintOffset }.paint();
        return;
    }

    if (box.isText()) {
        auto hasVisibleDamage = box.text().length() && box.isVisible() && hasDamage(box);
        if (!hasVisibleDamage)
            return;

        if (!box.layoutBox().rendererForIntegration()) {
            // FIXME: For some reason, we are getting to a state in which painting is requested for a box without renderer. We should try to figure out the root cause for this instead of bailing out here.
            ASSERT_NOT_REACHED();
            return;
        }

        TextBoxPainter { m_inlineContent, box, box.style(), m_paintInfo, m_paintOffset }.paint();
        return;
    }

    if (auto* renderer = dynamicDowncast<RenderBox>(box.layoutBox().rendererForIntegration()); renderer && renderer->isReplacedOrAtomicInline()) {
        if (m_paintInfo.shouldPaintWithinRoot(*renderer)) {
            // FIXME: Painting should not require a non-const renderer.
            const_cast<RenderBox*>(renderer)->paintAsInlineBlock(m_paintInfo, flippedContentOffsetIfNeeded(*renderer));
        }
    }
}

void InlineContentPainter::paint()
{
    auto layerPaintScope = LayerPaintScope { m_inlineBoxWithLayer };
    auto lastBoxLineIndex = std::optional<size_t> { };

    auto paintLineEndingEllipsisIfApplicable = [&](std::optional<size_t> currentLineIndex) {
        // Since line ending ellipsis belongs to the line structure but we don't have the concept of painting the line itself
        // let's paint it when we are either at the end of the content or finished painting a line with ellipsis.
        // While normally ellipsis is on the last line, -webkit-line-clamp can make us put ellipsis on any line.
        if (m_inlineBoxWithLayer) {
            // Line ending ellipsis is never on the inline box (with layer).
            return;
        }
        if (lastBoxLineIndex && (!currentLineIndex || *lastBoxLineIndex != currentLineIndex))
            paintEllipsis(*lastBoxLineIndex);
    };

    for (auto& box : m_inlineContent.boxesForRect(m_damageRect)) {
        auto shouldPaintBoxForPhase = [&] {
            switch (m_paintInfo.phase) {
            case PaintPhase::ChildOutlines:
                return box.isNonRootInlineBox();
            case PaintPhase::SelfOutline:
                return box.isRootInlineBox();
            case PaintPhase::Outline:
                return box.isInlineBox();
            case PaintPhase::Mask:
                return box.isInlineBox();
            default:
                return true;
            }
        };

        if (shouldPaintBoxForPhase() && layerPaintScope.includes(box)) {
            paintLineEndingEllipsisIfApplicable(box.lineIndex());
            paintDisplayBox(box);
        }
        lastBoxLineIndex = box.lineIndex();
    }
    paintLineEndingEllipsisIfApplicable({ });

    for (auto& renderInline : m_outlineObjects)
        renderInline.paintOutline(m_paintInfo, m_paintOffset);
}

LayoutPoint InlineContentPainter::flippedContentOffsetIfNeeded(const RenderBox& childRenderer) const
{
    if (root().writingMode().isBlockFlipped())
        return root().flipForWritingModeForChild(childRenderer, m_paintOffset);
    return m_paintOffset;
}

const RenderBlock& InlineContentPainter::root() const
{
    return m_root;
}

LayerPaintScope::LayerPaintScope(const RenderInline* inlineBoxWithLayer)
    : m_inlineBoxWithLayer(inlineBoxWithLayer ? inlineBoxWithLayer->layoutBox() : nullptr)
{
}

bool LayerPaintScope::includes(const InlineDisplay::Box& box)
{
    auto isInside = [](auto& displayBox, auto& inlineBox)
    {
        ASSERT(inlineBox.isInlineBox());

        if (displayBox.isRootInlineBox())
            return false;

        for (auto* box = &displayBox.layoutBox().parent(); box->isInlineBox(); box = &box->parent()) {
            if (box == &inlineBox)
                return true;
        }
        return false;
    };

    if (m_inlineBoxWithLayer == &box.layoutBox())
        return true;
    if (m_inlineBoxWithLayer && !isInside(box, *m_inlineBoxWithLayer))
        return false;
    if (m_currentExcludedInlineBox && isInside(box, *m_currentExcludedInlineBox))
        return false;

    m_currentExcludedInlineBox = nullptr;

    if (box.isRootInlineBox() || box.isText() || box.isLineBreak())
        return true;

    auto* renderer = dynamicDowncast<RenderLayerModelObject>(box.layoutBox().rendererForIntegration());
    bool hasSelfPaintingLayer = renderer && renderer->hasSelfPaintingLayer();

    if (hasSelfPaintingLayer && box.isNonRootInlineBox())
        m_currentExcludedInlineBox = &downcast<Layout::ElementBox>(box.layoutBox());

    return !hasSelfPaintingLayer;
}

}
}

