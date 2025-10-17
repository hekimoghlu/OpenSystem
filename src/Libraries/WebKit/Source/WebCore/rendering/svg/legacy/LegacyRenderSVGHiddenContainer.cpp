/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#include "LegacyRenderSVGHiddenContainer.h"

#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGHiddenContainer);

LegacyRenderSVGHiddenContainer::LegacyRenderSVGHiddenContainer(Type type, SVGElement& element, RenderStyle&& style, OptionSet<SVGModelObjectFlag> svgFlags)
    : LegacyRenderSVGContainer(type, element, WTFMove(style), svgFlags | SVGModelObjectFlag::IsHiddenContainer)
{
}

void LegacyRenderSVGHiddenContainer::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());
    SVGRenderSupport::layoutChildren(*this, selfNeedsLayout());
    clearNeedsLayout();
}

void LegacyRenderSVGHiddenContainer::paint(PaintInfo&, const LayoutPoint&)
{
    // This subtree does not paint.
}

void LegacyRenderSVGHiddenContainer::absoluteQuads(Vector<FloatQuad>&, bool*) const
{
    // This subtree does not take up space or paint
}

bool LegacyRenderSVGHiddenContainer::nodeAtFloatPoint(const HitTestRequest&, HitTestResult&, const FloatPoint&, HitTestAction)
{
    return false;
}

}
