/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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

#include "FormattingState.h"
#include "PlacedFloats.h"
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

// BlockFormattingState holds the state for a particular block formatting context tree.
class BlockFormattingState : public FormattingState {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BlockFormattingState);
public:
    BlockFormattingState(LayoutState&, const ElementBox& blockFormattingContextRoot);
    ~BlockFormattingState();

    const PlacedFloats& placedFloats() const { return m_placedFloats; }
    PlacedFloats& placedFloats() { return m_placedFloats; }

    // Since we layout the out-of-flow boxes at the end of the formatting context layout, it's okay to store them in the formatting state -as opposed to the containing block level.
    using OutOfFlowBoxList = Vector<CheckedRef<const Box>>;
    void addOutOfFlowBox(const Box& outOfFlowBox) { m_outOfFlowBoxes.append(outOfFlowBox); }
    void setOutOfFlowBoxes(OutOfFlowBoxList&& outOfFlowBoxes) { m_outOfFlowBoxes = WTFMove(outOfFlowBoxes); }
    const OutOfFlowBoxList& outOfFlowBoxes() const { return m_outOfFlowBoxes; }

    void setUsedVerticalMargin(const Box& layoutBox, const UsedVerticalMargin& usedVerticalMargin) { m_usedVerticalMargins.set(layoutBox, usedVerticalMargin); }
    UsedVerticalMargin usedVerticalMargin(const Box& layoutBox) const { return m_usedVerticalMargins.get(layoutBox); }
    bool hasUsedVerticalMargin(const Box& layoutBox) const { return m_usedVerticalMargins.contains(layoutBox); }

    void setHasClearance(const Box& layoutBox) { m_clearanceSet.add(layoutBox); }
    void clearHasClearance(const Box& layoutBox) { m_clearanceSet.remove(layoutBox); }
    bool hasClearance(const Box& layoutBox) const { return m_clearanceSet.contains(layoutBox); }

    void shrinkToFit();

private:
    PlacedFloats m_placedFloats;
    OutOfFlowBoxList m_outOfFlowBoxes;
    UncheckedKeyHashMap<CheckedRef<const Box>, UsedVerticalMargin> m_usedVerticalMargins;
    UncheckedKeyHashSet<CheckedRef<const Box>> m_clearanceSet;
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_STATE(BlockFormattingState, isBlockFormattingState())

