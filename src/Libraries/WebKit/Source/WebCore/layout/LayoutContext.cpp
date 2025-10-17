/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#include "LayoutContext.h"

#include "BlockFormattingContext.h"
#include "BlockFormattingState.h"
#include "LayoutBox.h"
#include "LayoutBoxGeometry.h"
#include "LayoutElementBox.h"
#include "LayoutPhase.h"
#include "LayoutTreeBuilder.h"
#include "RenderStyleConstants.h"
#include "RenderStyleSetters.h"
#include "RenderView.h"
#include "TableFormattingContext.h"
#include "TableFormattingState.h"
#include "TableWrapperBlockFormattingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LayoutContext);

LayoutContext::LayoutContext(LayoutState& layoutState)
    : m_layoutState(layoutState)
{
}

void LayoutContext::layout(const LayoutSize& rootContentBoxSize)
{
    // Set the geometry on the root.
    // Note that we never layout the root box. It has to have an already computed geometry (in case of ICB, it's the view geometry).
    // ICB establishes the initial BFC, but it does not live in a formatting context and while a non-ICB root(subtree layout) has to have a formatting context,
    // we could not lay it out even if we wanted to since it's outside of this LayoutContext.
    auto& boxGeometry = layoutState().geometryForRootBox();
    boxGeometry.setHorizontalMargin({ });
    boxGeometry.setVerticalMargin({ });
    boxGeometry.setBorder({ });
    boxGeometry.setPadding({ });
    boxGeometry.setTopLeft({ });
    boxGeometry.setContentBoxHeight(rootContentBoxSize.height());
    boxGeometry.setContentBoxWidth(rootContentBoxSize.width());

    auto scope = PhaseScope { Phase::Type::Layout };
    layoutFormattingContextSubtree(m_layoutState.root());
}


void LayoutContext::layoutFormattingContextSubtree(const ElementBox& formattingContextRoot)
{
    UNUSED_PARAM(formattingContextRoot);
}

std::unique_ptr<FormattingContext> LayoutContext::createFormattingContext(const ElementBox& formattingContextRoot, LayoutState& layoutState)
{
    ASSERT(formattingContextRoot.establishesFormattingContext());

    if (formattingContextRoot.establishesBlockFormattingContext()) {
        ASSERT(!formattingContextRoot.establishesInlineFormattingContext());
        auto& blockFormattingState = layoutState.ensureBlockFormattingState(formattingContextRoot);
        if (formattingContextRoot.isTableWrapperBox())
            return makeUnique<TableWrapperBlockFormattingContext>(formattingContextRoot, blockFormattingState);
        return makeUnique<BlockFormattingContext>(formattingContextRoot, blockFormattingState);
    }

    if (formattingContextRoot.establishesTableFormattingContext()) {
        auto& tableFormattingState = layoutState.ensureTableFormattingState(formattingContextRoot);
        return makeUnique<TableFormattingContext>(formattingContextRoot, tableFormattingState);
    }

    CRASH();
}

}
}

