/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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

#include "InlineContentCache.h"
#include "InlineItem.h"
#include "InlineLineTypes.h"
#include "LayoutElementBox.h"
#include "SecurityOrigin.h"
#include <wtf/HashMap.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
namespace Layout {
class InlineTextBox;

class InlineItemsBuilder {
public:
    InlineItemsBuilder(InlineContentCache&, const ElementBox& root, const SecurityOrigin&);
    void build(InlineItemPosition startPosition);

    static void populateBreakingPositionCache(const InlineItemList&, const Document&);

private:
    void collectInlineItems(InlineItemList&, InlineItemPosition startPosition);
    using LayoutQueue = Vector<CheckedRef<const Box>, 8>;
    LayoutQueue initializeLayoutQueue(InlineItemPosition startPosition);
    LayoutQueue traverseUntilDamaged(const Box& firstDamagedLayoutBox);
    void breakAndComputeBidiLevels(InlineItemList&);
    InlineContentCache::InlineItems::ContentAttributes computeContentAttributesAndInlineTextItemWidths(InlineItemList&, InlineItemPosition damagePosition, const InlineItemList& damagedItemList);

    void handleTextContent(const InlineTextBox&, InlineItemList&, std::optional<size_t> partialContentOffset);
    bool buildInlineItemListForTextFromBreakingPositionsCache(const InlineTextBox&, InlineItemList&);
    void handleInlineBoxStart(const Box&, InlineItemList&);
    void handleInlineBoxEnd(const Box&, InlineItemList&);
    void handleInlineLevelBox(const Box&, InlineItemList&);
    
    bool contentRequiresVisualReordering() const { return m_contentRequiresVisualReordering; }

    void computeInlineBoxBoundaryTextSpacings(const InlineItemList&);

    const ElementBox& root() const { return m_root; }
    InlineContentCache& inlineContentCache() { return m_inlineContentCache; }

private:
    InlineContentCache& m_inlineContentCache;
    const ElementBox& m_root;
    const SecurityOrigin& m_securityOrigin;

    bool m_contentRequiresVisualReordering { false };
    bool m_hasTextAutospace { !root().style().textAutospace().isNoAutospace() };
};

}
}

