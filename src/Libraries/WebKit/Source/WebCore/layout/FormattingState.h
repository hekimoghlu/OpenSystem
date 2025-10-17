/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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

#include "FormattingContext.h"
#include "LayoutState.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

class Box;
enum class StyleDiff;

class FormattingState {
    WTF_MAKE_NONCOPYABLE(FormattingState);
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FormattingState);
public:
    void setIntrinsicWidthConstraintsForBox(const Box&, IntrinsicWidthConstraints);
    std::optional<IntrinsicWidthConstraints> intrinsicWidthConstraintsForBox(const Box&) const;
    void clearIntrinsicWidthConstraints(const Box&);

    void setIntrinsicWidthConstraints(IntrinsicWidthConstraints intrinsicWidthConstraints) { m_intrinsicWidthConstraints = intrinsicWidthConstraints; }
    std::optional<IntrinsicWidthConstraints> intrinsicWidthConstraints() const { return m_intrinsicWidthConstraints; }

    bool isBlockFormattingState() const { return m_type == Type::Block; }
    bool isTableFormattingState() const { return m_type == Type::Table; }
    bool isFlexFormattingState() const { return m_type == Type::Flex; }

    LayoutState& layoutState() const { return m_layoutState; }

    // FIXME: We need to find a way to limit access to mutable geometry.
    BoxGeometry& boxGeometry(const Box& layoutBox);

protected:
    enum class Type { Block, Table, Flex };
    FormattingState(Type, LayoutState&);
    ~FormattingState();

private:
    LayoutState& m_layoutState;
    UncheckedKeyHashMap<const Box*, IntrinsicWidthConstraints> m_intrinsicWidthConstraintsForBoxes;
    std::optional<IntrinsicWidthConstraints> m_intrinsicWidthConstraints;
    Type m_type;
};

inline void FormattingState::setIntrinsicWidthConstraintsForBox(const Box& layoutBox, IntrinsicWidthConstraints intrinsicWidthConstraints)
{
    ASSERT(!m_intrinsicWidthConstraintsForBoxes.contains(&layoutBox));
    ASSERT(&m_layoutState.formattingStateForFormattingContext(FormattingContext::formattingContextRoot(layoutBox)) == this);
    m_intrinsicWidthConstraintsForBoxes.set(&layoutBox, intrinsicWidthConstraints);
}

inline void FormattingState::clearIntrinsicWidthConstraints(const Box& layoutBox)
{
    m_intrinsicWidthConstraints = { };
    m_intrinsicWidthConstraintsForBoxes.remove(&layoutBox);
}

inline std::optional<IntrinsicWidthConstraints> FormattingState::intrinsicWidthConstraintsForBox(const Box& layoutBox) const
{
    ASSERT(&m_layoutState.formattingStateForFormattingContext(FormattingContext::formattingContextRoot(layoutBox)) == this);
    auto iterator = m_intrinsicWidthConstraintsForBoxes.find(&layoutBox);
    if (iterator == m_intrinsicWidthConstraintsForBoxes.end())
        return { };
    return iterator->value;
}

}
}

#define SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_STATE(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Layout::ToValueTypeName) \
    static bool isType(const WebCore::Layout::FormattingState& formattingState) { return formattingState.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

