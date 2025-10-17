/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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

#include "AvailableLineWidthOverride.h"
#include "BlockLayoutState.h"

namespace WebCore {
namespace Layout {

class InlineLayoutState {
public:
    InlineLayoutState(BlockLayoutState&);

    void setClearGapAfterLastLine(InlineLayoutUnit verticalGap);
    InlineLayoutUnit clearGapAfterLastLine() const { return m_clearGapAfterLastLine; }

    void setClearGapBeforeFirstLine(InlineLayoutUnit verticalGap) { m_clearGapBeforeFirstLine = verticalGap; }
    InlineLayoutUnit clearGapBeforeFirstLine() const { return m_clearGapBeforeFirstLine; }

    const BlockLayoutState& parentBlockLayoutState() const { return m_parentBlockLayoutState; }
    BlockLayoutState& parentBlockLayoutState() { return m_parentBlockLayoutState; }

    const PlacedFloats& placedFloats() const { return m_parentBlockLayoutState.placedFloats(); }
    PlacedFloats& placedFloats() { return m_parentBlockLayoutState.placedFloats(); }

    void setAvailableLineWidthOverride(AvailableLineWidthOverride availableLineWidthOverride) { m_availableLineWidthOverride = availableLineWidthOverride; }
    const AvailableLineWidthOverride& availableLineWidthOverride() const { return m_availableLineWidthOverride; }

    void setLegacyClampedLineIndex(size_t lineIndex) { m_legacyClampedLineIndex = lineIndex; }
    std::optional<size_t> legacyClampedLineIndex() const { return m_legacyClampedLineIndex; }

    void setHyphenationLimitLines(size_t hyphenationLimitLines) { m_hyphenationLimitLines = hyphenationLimitLines; }
    void incrementSuccessiveHyphenatedLineCount() { ++m_successiveHyphenatedLineCount; }
    void resetSuccessiveHyphenatedLineCount() { m_successiveHyphenatedLineCount = 0; }
    bool isHyphenationDisabled() const { return m_hyphenationLimitLines && *m_hyphenationLimitLines <= m_successiveHyphenatedLineCount; }

    void setFirstLineStartTrimForInitialLetter(InlineLayoutUnit trimmedThisMuch) { m_firstLineStartTrimForInitialLetter = trimmedThisMuch; }
    InlineLayoutUnit firstLineStartTrimForInitialLetter() const { return m_firstLineStartTrimForInitialLetter; }

    void setInStandardsMode() { m_inStandardsMode = true; }
    bool inStandardsMode() const { return m_inStandardsMode; }

    // Integration codepath
    void setNestedListMarkerOffsets(UncheckedKeyHashMap<const ElementBox*, LayoutUnit>&& nestedListMarkerOffsets) { m_nestedListMarkerOffsets = WTFMove(nestedListMarkerOffsets); }
    LayoutUnit nestedListMarkerOffset(const ElementBox& listMarkerBox) const { return m_nestedListMarkerOffsets.get(&listMarkerBox); }
    void setShouldNotSynthesizeInlineBlockBaseline() { m_shouldNotSynthesizeInlineBlockBaseline = true; }
    bool shouldNotSynthesizeInlineBlockBaseline() const { return m_shouldNotSynthesizeInlineBlockBaseline; }

private:
    BlockLayoutState& m_parentBlockLayoutState;
    InlineLayoutUnit m_clearGapBeforeFirstLine { 0.f };
    InlineLayoutUnit m_clearGapAfterLastLine { 0.f };
    InlineLayoutUnit m_firstLineStartTrimForInitialLetter { 0.f };
    std::optional<size_t> m_legacyClampedLineIndex { };
    std::optional<size_t> m_hyphenationLimitLines { };
    size_t m_successiveHyphenatedLineCount { 0 };
    // FIXME: This is required by the integaration codepath.
    UncheckedKeyHashMap<const ElementBox*, LayoutUnit> m_nestedListMarkerOffsets;
    AvailableLineWidthOverride m_availableLineWidthOverride;
    bool m_shouldNotSynthesizeInlineBlockBaseline { false };
    bool m_inStandardsMode { false };
};

inline InlineLayoutState::InlineLayoutState(BlockLayoutState& parentBlockLayoutState)
    : m_parentBlockLayoutState(parentBlockLayoutState)
{
}

inline void InlineLayoutState::setClearGapAfterLastLine(InlineLayoutUnit verticalGap)
{
    ASSERT(verticalGap >= 0);
    m_clearGapAfterLastLine = verticalGap;
}

}
}
