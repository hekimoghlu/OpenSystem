/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#include "RenderSVGHiddenContainer.h"

#include "RenderLayer.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGHiddenContainer);

RenderSVGHiddenContainer::RenderSVGHiddenContainer(Type type, SVGElement& element, RenderStyle&& style, OptionSet<SVGModelObjectFlag> flags)
    : RenderSVGContainer(type, element, WTFMove(style), flags | SVGModelObjectFlag::IsHiddenContainer)
{
    ASSERT(isRenderSVGHiddenContainer());
}

RenderSVGHiddenContainer::~RenderSVGHiddenContainer() = default;

void RenderSVGHiddenContainer::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());

    layoutChildren();
    clearNeedsLayout();    
}

}
