/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

#include "FormattingConstraints.h"
#include "LayoutSize.h"

namespace WebCore {
namespace Layout {

struct ConstraintsForInlineContent : public ConstraintsForInFlowContent {
    ConstraintsForInlineContent(const ConstraintsForInFlowContent&, LayoutUnit visualLeft, LayoutSize containerRenderSize);

    LayoutUnit visualLeft() const { return m_visualLeft; }
    LayoutSize containerRenderSize() const { return m_containerRenderSize; }

private:
    LayoutUnit m_visualLeft;
    LayoutSize m_containerRenderSize;
};

inline ConstraintsForInlineContent::ConstraintsForInlineContent(const ConstraintsForInFlowContent& genericContraints, LayoutUnit visualLeft, LayoutSize containerRenderSize)
    : ConstraintsForInFlowContent(genericContraints.horizontal(), genericContraints.logicalTop(), InlineContent)
    , m_visualLeft(visualLeft)
    , m_containerRenderSize(containerRenderSize)
{
}

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_CONSTRAINTS(ConstraintsForInlineContent, isConstraintsForInlineContent())

