/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "config.h"
#include "RenderTreeBuilderBlockFlow.h"

#include "RenderMultiColumnFlow.h"
#include "RenderTreeBuilderBlock.h"
#include "RenderTreeBuilderMultiColumn.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderTreeBuilder::BlockFlow);

RenderTreeBuilder::BlockFlow::BlockFlow(RenderTreeBuilder& builder)
    : m_builder(builder)
{
}

void RenderTreeBuilder::BlockFlow::attach(RenderBlockFlow& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild)
{
    if (auto* multicolumnFlow = parent.multiColumnFlow()) {
        auto legendAvoidsMulticolumn = parent.isFieldset() && child->isLegend();
        if (legendAvoidsMulticolumn)
            return m_builder.blockBuilder().attach(parent, WTFMove(child), nullptr);

        auto legendBeforeChildIsIncorrect = parent.isFieldset() && beforeChild && beforeChild->isLegend();
        if (legendBeforeChildIsIncorrect)
            return m_builder.blockBuilder().attach(*multicolumnFlow, WTFMove(child), nullptr);

        // When the before child is set to be the first child of the RenderBlockFlow, we need to readjust it to be the first
        // child of the multicol conainter.
        return m_builder.attach(*multicolumnFlow, WTFMove(child), beforeChild == multicolumnFlow ? multicolumnFlow->firstChild() : beforeChild);
    }

    auto* beforeChildOrPlaceholder = beforeChild;
    if (auto* containingFragmentedFlow = parent.enclosingFragmentedFlow())
        beforeChildOrPlaceholder = m_builder.multiColumnBuilder().resolveMovedChild(*containingFragmentedFlow, beforeChild);
    m_builder.blockBuilder().attach(parent, WTFMove(child), beforeChildOrPlaceholder);
}

void RenderTreeBuilder::BlockFlow::moveAllChildrenIncludingFloats(RenderBlockFlow& from, RenderBlock& to, RenderTreeBuilder::NormalizeAfterInsertion normalizeAfterInsertion)
{
    m_builder.moveAllChildren(from, to, normalizeAfterInsertion);
    from.addFloatsToNewParent(downcast<RenderBlockFlow>(to));
}

}
