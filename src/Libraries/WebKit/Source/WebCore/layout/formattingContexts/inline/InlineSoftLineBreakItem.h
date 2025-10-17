/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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

#include "InlineItem.h"

namespace WebCore {
namespace Layout {

class InlineSoftLineBreakItem : public InlineItem {
public:
    static InlineSoftLineBreakItem createSoftLineBreakItem(const InlineTextBox&, unsigned position, UBiDiLevel = UBIDI_DEFAULT_LTR);

    unsigned position() const { return m_startOrPosition; }
    const InlineTextBox& inlineTextBox() const { return downcast<InlineTextBox>(layoutBox()); }

private:
    InlineSoftLineBreakItem(const InlineTextBox&, unsigned position, UBiDiLevel);
};

inline InlineSoftLineBreakItem InlineSoftLineBreakItem::createSoftLineBreakItem(const InlineTextBox& inlineTextBox, unsigned position, UBiDiLevel bidiLevel)
{
    return { inlineTextBox, position, bidiLevel };
}

inline InlineSoftLineBreakItem::InlineSoftLineBreakItem(const InlineTextBox& inlineTextBox, unsigned position, UBiDiLevel bidiLevel)
    : InlineItem(inlineTextBox, Type::SoftLineBreak, bidiLevel)
{
    m_startOrPosition = position;
}

}
}

SPECIALIZE_TYPE_TRAITS_INLINE_ITEM(InlineSoftLineBreakItem, isSoftLineBreak())

