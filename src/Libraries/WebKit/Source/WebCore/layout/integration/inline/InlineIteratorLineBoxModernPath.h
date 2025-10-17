/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

#include "InlineIteratorBoxModernPath.h"
#include "LayoutIntegrationInlineContent.h"
#include "RenderBlockFlow.h"

namespace WebCore {
namespace InlineIterator {

class BoxModernPath;

class LineBoxIteratorModernPath {
public:
    LineBoxIteratorModernPath(const LayoutIntegration::InlineContent& inlineContent, size_t lineIndex)
        : m_inlineContent(&inlineContent)
        , m_lineIndex(lineIndex)
    {
        ASSERT(lineIndex <= lines().size());
    }
    LineBoxIteratorModernPath(LineBoxIteratorModernPath&&) = default;
    LineBoxIteratorModernPath(const LineBoxIteratorModernPath&) = default;
    LineBoxIteratorModernPath& operator=(const LineBoxIteratorModernPath&) = default;
    LineBoxIteratorModernPath& operator=(LineBoxIteratorModernPath&&) = default;

    float contentLogicalTop() const { return line().enclosingContentLogicalTop(); }
    float contentLogicalBottom() const { return line().enclosingContentLogicalBottom(); }
    float logicalTop() const { return line().lineBoxLogicalRect().y(); }
    float logicalBottom() const { return line().lineBoxLogicalRect().maxY(); }
    float logicalWidth() const { return line().lineBoxLogicalRect().width(); }
    float inkOverflowLogicalTop() const { return line().isHorizontal() ? line().inkOverflow().y() : line().inkOverflow().x(); }
    float inkOverflowLogicalBottom() const { return line().isHorizontal() ? line().inkOverflow().maxY() : line().inkOverflow().maxX(); }
    float scrollableOverflowTop() const { return line().scrollableOverflow().y(); }
    float scrollableOverflowBottom() const { return line().scrollableOverflow().maxY(); }

    bool hasEllipsis() const { return line().hasEllipsis(); }
    FloatRect ellipsisVisualRectIgnoringBlockDirection() const { return line().ellipsis()->visualRect; }
    TextRun ellipsisText() const { return TextRun { line().ellipsis()->text.string() }; }

    float contentLogicalTopAdjustedForPrecedingLineBox() const
    {
        if (formattingContextRoot().writingMode().isLineInverted() || !m_lineIndex)
            return contentLogicalTop();
        return LineBoxIteratorModernPath { *m_inlineContent, m_lineIndex - 1 }.contentLogicalBottom();
    }
    float contentLogicalBottomAdjustedForFollowingLineBox() const
    {
        if (!formattingContextRoot().writingMode().isLineInverted() || m_lineIndex == lines().size() - 1)
            return contentLogicalBottom();
        return LineBoxIteratorModernPath { *m_inlineContent, m_lineIndex + 1 }.contentLogicalTop();
    }

    float contentLogicalLeft() const
    {
        auto writingMode = formattingContextRoot().writingMode();
        if (writingMode.isLogicalLeftLineLeft())
            return line().lineBoxLeft() + line().contentLogicalLeftIgnoringInlineDirection();
        ASSERT(writingMode.isVertical()); // Currently only sideways-lr gets this far.
        return line().bottom() - (line().contentLogicalLeftIgnoringInlineDirection() + line().contentLogicalWidth());
    }
    float contentLogicalRight() const { return contentLogicalLeft() + line().contentLogicalWidth(); }
    bool isHorizontal() const { return line().isHorizontal(); }
    FontBaseline baselineType() const { return line().baselineType(); }

    const RenderBlockFlow& formattingContextRoot() const { return m_inlineContent->formattingContextRoot(); }

    bool isFirstAfterPageBreak() const { return line().isFirstAfterPageBreak(); }

    size_t lineIndex() const { return m_lineIndex; }

    void traverseNext()
    {
        ASSERT(!atEnd());

        ++m_lineIndex;
    }

    void traversePrevious()
    {
        ASSERT(!atEnd());

        if (!m_lineIndex) {
            setAtEnd();
            return;
        }

        --m_lineIndex;
    }

    friend bool operator==(const LineBoxIteratorModernPath&, const LineBoxIteratorModernPath&) = default;

    bool atEnd() const { return !m_inlineContent || m_lineIndex == lines().size(); }

    BoxModernPath firstLeafBox() const
    {
        if (!line().boxCount())
            return { *m_inlineContent };
        auto runIterator = BoxModernPath { *m_inlineContent, line().firstBoxIndex() };
        if (runIterator.box().isInlineBox())
            runIterator.traverseNextLeafOnLine();
        return runIterator;
    }

    BoxModernPath lastLeafBox() const
    {
        auto boxCount = line().boxCount();
        if (!boxCount)
            return { *m_inlineContent };
        auto runIterator = BoxModernPath { *m_inlineContent, line().firstBoxIndex() + boxCount - 1 };
        if (runIterator.box().isInlineBox())
            runIterator.traversePreviousLeafOnLine();
        return runIterator;
    }

private:
    void setAtEnd() { m_lineIndex = lines().size(); }

    const InlineDisplay::Lines& lines() const { return m_inlineContent->displayContent().lines; }
    const InlineDisplay::Line& line() const { return lines()[m_lineIndex]; }

    WeakPtr<const LayoutIntegration::InlineContent> m_inlineContent;
    size_t m_lineIndex { 0 };
};

}
}

