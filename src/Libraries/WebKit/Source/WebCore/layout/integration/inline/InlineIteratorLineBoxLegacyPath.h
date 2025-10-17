/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

#include "InlineIteratorBoxLegacyPath.h"
#include "LayoutIntegrationInlineContent.h"
#include "LegacyRootInlineBox.h"
#include "RenderBlockFlow.h"

namespace WebCore {
namespace InlineIterator {

class LineBoxIteratorLegacyPath {
public:
    LineBoxIteratorLegacyPath(const LegacyRootInlineBox* rootInlineBox)
        : m_rootInlineBox(rootInlineBox)
    {
    }
    LineBoxIteratorLegacyPath(LineBoxIteratorLegacyPath&&) = default;
    LineBoxIteratorLegacyPath(const LineBoxIteratorLegacyPath&) = default;
    LineBoxIteratorLegacyPath& operator=(const LineBoxIteratorLegacyPath&) = default;
    LineBoxIteratorLegacyPath& operator=(LineBoxIteratorLegacyPath&&) = default;

    float contentLogicalTop() const { return m_rootInlineBox->lineTop().toFloat(); }
    float contentLogicalBottom() const { return m_rootInlineBox->lineBottom().toFloat(); }
    float contentLogicalTopAdjustedForPrecedingLineBox() const { return m_rootInlineBox->selectionTop().toFloat(); }
    float contentLogicalBottomAdjustedForFollowingLineBox() const { return m_rootInlineBox->selectionBottom().toFloat(); }
    float logicalTop() const { return m_rootInlineBox->lineBoxTop().toFloat(); }
    float logicalBottom() const { return m_rootInlineBox->lineBoxBottom().toFloat(); }
    float logicalWidth() const { return m_rootInlineBox->lineBoxWidth().toFloat(); }
    float inkOverflowLogicalTop() const { return m_rootInlineBox->logicalTopVisualOverflow(); }
    float inkOverflowLogicalBottom() const { return m_rootInlineBox->logicalBottomVisualOverflow(); }
    float scrollableOverflowTop() const { return logicalTop(); }
    float scrollableOverflowBottom() const { return logicalBottom(); }

    bool hasEllipsis() const { return false; }
    FloatRect ellipsisVisualRectIgnoringBlockDirection() const
    {
        ASSERT_NOT_REACHED();
        return { };
    }

    TextRun ellipsisText() const
    {
        ASSERT_NOT_REACHED();
        return TextRun { emptyString() };
    }

    float contentLogicalLeft() const { return m_rootInlineBox->logicalLeft(); }
    float contentLogicalRight() const { return m_rootInlineBox->logicalRight(); }
    bool isHorizontal() const { return m_rootInlineBox->isHorizontal(); }
    FontBaseline baselineType() const { return m_rootInlineBox->baselineType(); }

    const RenderBlockFlow& formattingContextRoot() const { return m_rootInlineBox->blockFlow(); }

    bool isFirstAfterPageBreak() const { return false; }

    size_t lineIndex() const
    {
        return formattingContextRoot().legacyRootBox() ? 1 : 0;
    }


    void traverseNext()
    {
        m_rootInlineBox = m_rootInlineBox->nextRootBox();
    }

    void traversePrevious()
    {
        m_rootInlineBox = m_rootInlineBox->prevRootBox();
    }

    friend bool operator==(LineBoxIteratorLegacyPath, LineBoxIteratorLegacyPath) = default;

    bool atEnd() const { return !m_rootInlineBox; }

    BoxLegacyPath firstLeafBox() const
    {
        return { m_rootInlineBox->firstLeafDescendant() };
    }

    BoxLegacyPath lastLeafBox() const
    {
        return { m_rootInlineBox->lastLeafDescendant() };
    }

private:
    WeakPtr<const LegacyRootInlineBox> m_rootInlineBox;
};

}
}
