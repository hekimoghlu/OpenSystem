/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include "InlineIteratorLineBox.h"
#include "InlineIteratorLineBoxInlines.h"

#include "InlineIteratorBoxInlines.h"
#include "LayoutIntegrationLineLayout.h"
#include "RenderBlockFlow.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"

namespace WebCore {
namespace InlineIterator {

LineBoxIterator::LineBoxIterator(LineBox::PathVariant&& pathVariant)
    : m_lineBox(WTFMove(pathVariant))
{
}

LineBoxIterator::LineBoxIterator(const LineBox& lineBox)
    : m_lineBox(lineBox)
{
}

bool LineBoxIterator::atEnd() const
{
    return WTF::switchOn(m_lineBox.m_pathVariant, [](auto& path) {
        return path.atEnd();
    });
}

LineBoxIterator& LineBoxIterator::traverseNext()
{
    WTF::switchOn(m_lineBox.m_pathVariant, [](auto& path) {
        return path.traverseNext();
    });
    return *this;
}

LineBoxIterator& LineBoxIterator::traversePrevious()
{
    WTF::switchOn(m_lineBox.m_pathVariant, [](auto& path) {
        return path.traversePrevious();
    });
    return *this;
}

LineBoxIterator::operator bool() const
{
    return !atEnd();
}

bool LineBoxIterator::operator==(const LineBoxIterator& other) const
{
    return m_lineBox.m_pathVariant == other.m_lineBox.m_pathVariant;
}

LineBoxIterator firstLineBoxFor(const RenderBlockFlow& flow)
{
    if (auto* lineLayout = flow.inlineLayout())
        return lineLayout->firstLineBox();

    return { LineBoxIteratorLegacyPath { flow.legacyRootBox() } };
}

LineBoxIterator lastLineBoxFor(const RenderBlockFlow& flow)
{
    if (auto* lineLayout = flow.inlineLayout())
        return lineLayout->lastLineBox();

    return { LineBoxIteratorLegacyPath { flow.legacyRootBox() } };
}

LineBoxIterator lineBoxFor(const LayoutIntegration::InlineContent& inlineContent, size_t lineIndex)
{
    return { LineBoxIteratorModernPath { inlineContent, lineIndex } };
}


LineBoxIterator LineBox::next() const
{
    return LineBoxIterator(*this).traverseNext();
}

LineBoxIterator LineBox::previous() const
{
    return LineBoxIterator(*this).traversePrevious();
}

LeafBoxIterator LineBox::lineLeftmostLeafBox() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> LeafBoxIterator {
        return { path.firstLeafBox() };
    });
}

LeafBoxIterator LineBox::lineRightmostLeafBox() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> LeafBoxIterator {
        return { path.lastLeafBox() };
    });
}

LeafBoxIterator closestBoxForHorizontalPosition(const LineBox& lineBox, float horizontalPosition, bool editableOnly)
{
    auto isEditable = [&](auto box) {
        return box && box->renderer().node() && box->renderer().node()->hasEditableStyle();
    };

    auto firstBox = lineBox.logicalLeftmostLeafBox();
    auto lastBox = lineBox.logicalRightmostLeafBox();

    if (firstBox != lastBox) {
        if (firstBox->isLineBreak())
            firstBox = firstBox->nextLogicalRightwardOnLineIgnoringLineBreak();
        else if (lastBox->isLineBreak())
            lastBox = lastBox->nextLogicalLeftwardOnLineIgnoringLineBreak();
    }

    if (firstBox == lastBox && (!editableOnly || isEditable(firstBox)))
        return firstBox;

    if (firstBox && horizontalPosition <= firstBox->logicalLeft() && !firstBox->renderer().isRenderListMarker() && (!editableOnly || isEditable(firstBox)))
        return firstBox;

    if (lastBox && horizontalPosition >= lastBox->logicalRight() && !lastBox->renderer().isRenderListMarker() && (!editableOnly || isEditable(lastBox)))
        return lastBox;

    auto closestBox = lastBox;
    for (auto box = firstBox; box; box = box.traverseLogicalRightwardOnLineIgnoringLineBreak()) {
        if (!box->renderer().isRenderListMarker() && (!editableOnly || isEditable(box))) {
            if (horizontalPosition < box->logicalRight())
                return box;
            closestBox = box;
        }
    }

    return closestBox;
}

RenderObject::HighlightState LineBox::ellipsisSelectionState() const
{
    auto lastLeafBox = this->lineRightmostLeafBox();
    if (!lastLeafBox)
        return RenderObject::HighlightState::None;

    auto* text = dynamicDowncast<InlineIterator::TextBox>(*lastLeafBox);
    if (!text || text->selectionState() == RenderObject::HighlightState::None)
        return RenderObject::HighlightState::None;

    auto selectionRange = text->selectableRange();
    if (!selectionRange.truncation)
        return RenderObject::HighlightState::None;

    auto [selectionStart, selectionEnd] = formattingContextRoot().view().selection().rangeForTextBox(text->renderer(), selectionRange);
    return selectionStart <= *selectionRange.truncation && selectionEnd >= *selectionRange.truncation ? RenderObject::HighlightState::Inside : RenderObject::HighlightState::None;
}

}
}

