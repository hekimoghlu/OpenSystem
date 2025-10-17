/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#include "InlineIteratorTextBox.h"

#include "InlineIteratorBoxInlines.h"
#include "InlineIteratorLineBox.h"
#include "InlineIteratorTextBoxInlines.h"
#include "LayoutIntegrationLineLayout.h"
#include "RenderCombineText.h"
#include "RenderStyleInlines.h"
#include "SVGInlineTextBox.h"

namespace WebCore {
namespace InlineIterator {

TextBoxIterator TextBox::nextTextBox() const
{
    return TextBoxIterator(*this).traverseNextTextBox();
}

const FontCascade& TextBox::fontCascade() const
{
    if (auto* renderer = dynamicDowncast<RenderCombineText>(this->renderer()); renderer && renderer->isCombined())
        return renderer->textCombineFont();

    return style().fontCascade();
}

TextBoxIterator::TextBoxIterator(Box::PathVariant&& pathVariant)
    : LeafBoxIterator(WTFMove(pathVariant))
{
}

TextBoxIterator::TextBoxIterator(const Box& box)
    : LeafBoxIterator(box)
{
}

TextBoxIterator& TextBoxIterator::traverseNextTextBox()
{
    WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        path.traverseNextTextBox();
    });
    return *this;
}

TextBoxIterator lineLeftmostTextBoxFor(const RenderText& text)
{
    if (auto* lineLayout = LayoutIntegration::LineLayout::containing(text))
        return lineLayout->textBoxesFor(text);

    return { BoxLegacyPath { text.firstLegacyTextBox() } };
}

TextBoxIterator textBoxFor(const LegacyInlineTextBox* legacyInlineTextBox)
{
    return { BoxLegacyPath { legacyInlineTextBox } };
}

TextBoxIterator textBoxFor(const LayoutIntegration::InlineContent& content, const InlineDisplay::Box& box)
{
    return textBoxFor(content, content.indexForBox(box));
}

TextBoxIterator textBoxFor(const LayoutIntegration::InlineContent& content, size_t boxIndex)
{
    ASSERT(content.displayContent().boxes[boxIndex].isTextOrSoftLineBreak());
    return { BoxModernPath { content, boxIndex } };
}

BoxRange<TextBoxIterator> textBoxesFor(const RenderText& text)
{
    return { lineLeftmostTextBoxFor(text) };
}

}
}
