/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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

namespace WebCore {
namespace Layout {

struct ConstraintsForTableContent : public ConstraintsForInFlowContent {
    ConstraintsForTableContent(const ConstraintsForInFlowContent&, std::optional<LayoutUnit> availableVerticalSpaceForContent);

    std::optional<LayoutUnit> availableVerticalSpaceForContent() const { return m_availableVerticalSpaceForContent; }

private:
    std::optional<LayoutUnit> m_availableVerticalSpaceForContent;
};

inline ConstraintsForTableContent::ConstraintsForTableContent(const ConstraintsForInFlowContent& inFlowContraints, std::optional<LayoutUnit> availableVerticalSpaceForContent)
    : ConstraintsForInFlowContent(inFlowContraints.horizontal(), inFlowContraints.logicalTop(), TableContent)
    , m_availableVerticalSpaceForContent(availableVerticalSpaceForContent)
{
}

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_CONSTRAINTS(ConstraintsForTableContent, isConstraintsForTableContent())

