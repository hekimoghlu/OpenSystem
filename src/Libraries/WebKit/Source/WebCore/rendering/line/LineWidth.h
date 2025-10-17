/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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

#include "LayoutUnit.h"

namespace WebCore {

class FloatingObject;
class RenderBlockFlow;
class RenderObject;
class RenderStyle;

struct LineSegment;

class LineWidth {
public:
    LineWidth(RenderBlockFlow&, bool isFirstLine);

    bool fitsOnLine(bool ignoringTrailingSpace = false) const;
    bool fitsOnLineIncludingExtraWidth(float extra) const;
    bool fitsOnLineExcludingTrailingWhitespace(float extra) const;

    float currentWidth() const { return m_committedWidth + m_uncommittedWidth; }
    // FIXME: We should eventually replace these three functions by ones that work on a higher abstraction.
    float uncommittedWidth() const { return m_uncommittedWidth; }
    float committedWidth() const { return m_committedWidth; }
    float availableWidth() const { return m_availableWidth; }
    float logicalLeftOffset() const { return m_left; }
    
    bool hasCommitted() const { return m_hasCommitted; }
    bool hasCommittedReplaced() const { return m_hasCommittedReplaced; }

    void updateAvailableWidth(LayoutUnit minimumHeight = 0_lu);
    void shrinkAvailableWidthForNewFloatIfNeeded(const FloatingObject&);
    void addUncommittedWidth(float delta)
    {
        m_uncommittedWidth += delta;
    }
    void addUncommittedReplacedWidth(float delta)
    {
        addUncommittedWidth(delta);
        m_hasUncommittedReplaced = true;
    }
    void commit();
    void fitBelowFloats(bool isFirstLine = false);
    void setTrailingWhitespaceWidth(float collapsedWhitespace, float borderPaddingMargin = 0);

    bool isFirstLine() const { return m_isFirstLine; }

private:
    void computeAvailableWidthFromLeftAndRight();
    bool fitsOnLineExcludingTrailingCollapsedWhitespace() const;
    void updateLineDimension(LayoutUnit newLineTop, LayoutUnit newLineWidth, float newLineLeft, float newLineRight);
    void wrapNextToShapeOutside(bool isFirstLine);

    RenderBlockFlow& m_block;
    float m_uncommittedWidth { 0 };
    float m_committedWidth { 0 };
    float m_trailingWhitespaceWidth { 0 };
    float m_trailingCollapsedWhitespaceWidth { 0 };
    float m_left { 0 };
    float m_right { 0 };
    float m_availableWidth { 0 };
    bool m_isFirstLine { true };
    bool m_hasCommitted { false };
    bool m_hasCommittedReplaced { false };
    bool m_hasUncommittedReplaced { false };
};

} // namespace WebCore
