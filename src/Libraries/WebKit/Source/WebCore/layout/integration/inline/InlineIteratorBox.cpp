/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
#include "InlineIteratorBox.h"

#include "InlineIteratorInlineBox.h"
#include "InlineIteratorLineBox.h"
#include "InlineIteratorTextBox.h"
#include "LayoutIntegrationLineLayout.h"
#include "RenderBlockFlow.h"
#include "RenderLineBreak.h"
#include "RenderView.h"

namespace WebCore {
namespace InlineIterator {

BoxIterator::BoxIterator(Box::PathVariant&& pathVariant)
    : m_box(WTFMove(pathVariant))
{
}

BoxIterator::BoxIterator(const Box& run)
    : m_box(run)
{
}

bool BoxIterator::operator==(const BoxIterator& other) const
{
    if (atEnd() && other.atEnd())
        return true;

    return m_box.m_pathVariant == other.m_box.m_pathVariant;
}

bool BoxIterator::atEnd() const
{
    return WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        return path.atEnd();
    });
}

BoxIterator& BoxIterator::traverseLineRightwardOnLine()
{
    WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        path.traverseNextBoxOnLine();
    });
    return *this;
}

BoxIterator& BoxIterator::traverseLineRightwardOnLineSkippingChildren()
{
    WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        path.traverseNextBoxOnLineSkippingChildren();
    });
    return *this;
}

bool Box::isSVGText() const
{
    return isText() && renderer().isRenderSVGInlineText();
}

LeafBoxIterator Box::nextLineRightwardOnLine() const
{
    return LeafBoxIterator(*this).traverseLineRightwardOnLine();
}

LeafBoxIterator Box::nextLineLeftwardOnLine() const
{
    return LeafBoxIterator(*this).traverseLineLeftwardOnLine();
}

LeafBoxIterator Box::nextLineRightwardOnLineIgnoringLineBreak() const
{
    return LeafBoxIterator(*this).traverseLineRightwardOnLineIgnoringLineBreak();
}

LeafBoxIterator Box::nextLineLeftwardOnLineIgnoringLineBreak() const
{
    return LeafBoxIterator(*this).traverseLineLeftwardOnLineIgnoringLineBreak();
}

InlineBoxIterator Box::parentInlineBox() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> InlineBoxIterator {
        return { path.parentInlineBox() };
    });
}

LineBoxIterator Box::lineBox() const
{
    return WTF::switchOn(m_pathVariant, [](const BoxLegacyPath& path) {
        return LineBoxIterator(LineBoxIteratorLegacyPath(&path.rootInlineBox()));
    }
    , [](const BoxModernPath& path) {
        return LineBoxIterator(LineBoxIteratorModernPath(path.inlineContent(), path.box().lineIndex()));
    }
    );
}

FloatRect Box::visualRect() const
{
    auto rect = visualRectIgnoringBlockDirection();
    formattingContextRoot().flipForWritingMode(rect);
    return rect;
}

RenderObject::HighlightState Box::selectionState() const
{
    if (!hasRenderer())
        return { };

    if (auto* text = dynamicDowncast<TextBox>(*this)) {
        auto& renderer = text->renderer();
        return renderer.view().selection().highlightStateForTextBox(renderer, text->selectableRange());
    }
    return renderer().selectionState();
}

LeafBoxIterator::LeafBoxIterator(Box::PathVariant&& pathVariant)
    : BoxIterator(WTFMove(pathVariant))
{
}

LeafBoxIterator::LeafBoxIterator(const Box& run)
    : BoxIterator(run)
{
}

LeafBoxIterator& LeafBoxIterator::traverseLineRightwardOnLine()
{
    WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        path.traverseNextLeafOnLine();
    });
    return *this;
}

LeafBoxIterator& LeafBoxIterator::traverseLineLeftwardOnLine()
{
    WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        path.traversePreviousLeafOnLine();
    });
    return *this;
}

LeafBoxIterator& LeafBoxIterator::traverseLineRightwardOnLineIgnoringLineBreak()
{
    do {
        traverseLineRightwardOnLine();
    } while (!atEnd() && m_box.isLineBreak());
    return *this;
}

LeafBoxIterator& LeafBoxIterator::traverseLineLeftwardOnLineIgnoringLineBreak()
{
    do {
        traverseLineLeftwardOnLine();
    } while (!atEnd() && m_box.isLineBreak());
    return *this;
}

LeafBoxIterator boxFor(const RenderLineBreak& renderer)
{
    if (auto* lineLayout = LayoutIntegration::LineLayout::containing(renderer))
        return lineLayout->boxFor(renderer);
    return { };
}

LeafBoxIterator boxFor(const RenderBox& renderer)
{
    if (auto* lineLayout = LayoutIntegration::LineLayout::containing(renderer))
        return lineLayout->boxFor(renderer);
    return { };
}

LeafBoxIterator boxFor(const LayoutIntegration::InlineContent& content, size_t boxIndex)
{
    return { BoxModernPath { content, boxIndex } };
}

const BoxModernPath& Box::modernPath() const
{
    return std::get<BoxModernPath>(m_pathVariant);
}

const BoxLegacyPath& Box::legacyPath() const
{
    return std::get<BoxLegacyPath>(m_pathVariant);
}

}
}
