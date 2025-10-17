/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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

#include "InlineDisplayContent.h"
#include "InlineLineTypes.h"
#include <wtf/OptionSet.h>

namespace WebCore {
namespace Layout {

class Box;
class InlineInvalidation;

class InlineDamage {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InlineDamage);
public:
    InlineDamage() = default;
    ~InlineDamage();

    enum class Reason : uint8_t {
        Append                 = 1 << 0,
        Insert                 = 1 << 1,
        Remove                 = 1 << 2,
        ContentChange          = 1 << 3,
        StyleChange            = 1 << 4,
        Pagination             = 1 << 5
    };
    OptionSet<Reason> reasons() const { return m_damageReasons; }

    // FIXME: Add support for damage range with multiple, different damage types.
    struct LayoutPosition {
        size_t lineIndex { 0 };
        InlineItemPosition inlineItemPosition { };
        LayoutUnit partialContentTop;
    };
    std::optional<LayoutPosition> layoutStartPosition() const { return m_layoutStartPosition; }

    using TrailingDisplayBoxList = Vector<InlineDisplay::Box>;
    std::optional<InlineDisplay::Box> trailingContentForLine(size_t lineIndex) const;

    void addDetachedBox(UniqueRef<Box>&& layoutBox) { m_detachedLayoutBoxes.append(WTFMove(layoutBox)); }

    bool isInlineItemListDirty() const { return m_isInlineItemListDirty; }
    void setInlineItemListClean() { m_isInlineItemListDirty = false; }

#ifndef NDEBUG
    bool hasDetachedContent() const { return !m_detachedLayoutBoxes.isEmpty(); }
#endif

private:
    friend class InlineInvalidation;

    void setDamageReason(Reason reason) { m_damageReasons.add(reason); }
    void setLayoutStartPosition(LayoutPosition position) { m_layoutStartPosition = position; }
    void resetLayoutPosition();
    void setTrailingDisplayBoxes(TrailingDisplayBoxList&& trailingDisplayBoxes) { m_trailingDisplayBoxes = WTFMove(trailingDisplayBoxes); }
    void setInlineItemListDirty() { m_isInlineItemListDirty = true; }

    OptionSet<Reason> m_damageReasons;
    bool m_isInlineItemListDirty { false };
    std::optional<LayoutPosition> m_layoutStartPosition;
    TrailingDisplayBoxList m_trailingDisplayBoxes;
    Vector<UniqueRef<Box>> m_detachedLayoutBoxes;
};

inline std::optional<InlineDisplay::Box> InlineDamage::trailingContentForLine(size_t lineIndex) const
{
    if (m_trailingDisplayBoxes.isEmpty()) {
        // Couldn't compute trailing positions for damaged lines.
        return { };
    }
    if (!layoutStartPosition() || layoutStartPosition()->lineIndex > lineIndex) {
        ASSERT_NOT_REACHED();
        return { };
    }
    auto relativeLineIndex = lineIndex - layoutStartPosition()->lineIndex;
    if (relativeLineIndex >= m_trailingDisplayBoxes.size()) {
        // At the time of the damage, we didn't have this line yet -e.g content insert at a new line.
        return { };
    }
    return { m_trailingDisplayBoxes[relativeLineIndex] };
}

inline InlineDamage::~InlineDamage()
{
    m_trailingDisplayBoxes.clear();
    m_detachedLayoutBoxes.clear();
}

inline void InlineDamage::resetLayoutPosition()
{
    m_layoutStartPosition = { };
    m_trailingDisplayBoxes.clear();
    // Never reset m_detachedLayoutBoxes. We need to keep those layout boxes around until after layout.
}

}
}
