/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#include "RenderTableCaption.h"

#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderTable.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderTableCaption);

RenderTableCaption::RenderTableCaption(Element& element, RenderStyle&& style)
    : RenderBlockFlow(Type::TableCaption, element, WTFMove(style))
{
    ASSERT(isRenderTableCaption());
}

RenderTableCaption::~RenderTableCaption() = default;

void RenderTableCaption::insertedIntoTree()
{
    RenderBlockFlow::insertedIntoTree();
    table()->addCaption(*this);
}

void RenderTableCaption::willBeRemovedFromTree()
{
    RenderBlockFlow::willBeRemovedFromTree();
    table()->removeCaption(*this);
}

RenderTable* RenderTableCaption::table() const
{
    return downcast<RenderTable>(parent());
}

LayoutUnit RenderTableCaption::containingBlockLogicalWidthForContent() const
{
    if (auto* containingBlock = this->containingBlock())
        return containingBlock->logicalWidth();
    return { };
}

}
