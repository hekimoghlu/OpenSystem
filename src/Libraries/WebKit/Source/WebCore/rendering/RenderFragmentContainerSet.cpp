/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#include "RenderFragmentContainerSet.h"

#include "RenderBoxFragmentInfo.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderFragmentedFlow.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderFragmentContainerSet);

RenderFragmentContainerSet::RenderFragmentContainerSet(Type type, Document& document, RenderStyle&& style, RenderFragmentedFlow& fragmentedFlow)
    : RenderFragmentContainer(type, document, WTFMove(style), &fragmentedFlow)
{
    ASSERT(is<RenderFragmentContainerSet>(*this));
}

RenderFragmentContainerSet::~RenderFragmentContainerSet() = default;

void RenderFragmentContainerSet::installFragmentedFlow()
{
    // We don't have to do anything, since we were able to connect the flow thread
    // in the constructor.
}

void RenderFragmentContainerSet::expandToEncompassFragmentedFlowContentsIfNeeded()
{
    // Whenever the last region is a set, it always expands its region rect to consume all
    // of the flow thread content. This is because it is always capable of generating an
    // infinite number of boxes in order to hold all of the remaining content.
    auto rect = fragmentedFlowPortionRect();
    
    // Get the offset within the flow thread in its block progression direction. Then get the
    // flow thread's remaining logical height including its overflow and expand our rect
    // to encompass that remaining height and overflow. The idea is that we will generate
    // additional columns and pages to hold that overflow, since people do write bad
    // content like <body style="height:0px"> in multi-column layouts.
    bool isHorizontal = fragmentedFlow()->isHorizontalWritingMode();
    auto logicalTopOffset = isHorizontal ? rect.y() : rect.x();
    auto overflowHeight = isHorizontal ? fragmentedFlow()->layoutOverflowRect().maxY() : fragmentedFlow()->layoutOverflowRect().maxX();
    auto logicalHeightWithOverflow = logicalTopOffset == RenderFragmentedFlow::maxLogicalHeight() ? overflowHeight : overflowHeight - logicalTopOffset;
    setFragmentedFlowPortionRect({ rect.x(), rect.y(), isHorizontal ? rect.width() : logicalHeightWithOverflow, isHorizontal ? logicalHeightWithOverflow : rect.height() });
}

}
