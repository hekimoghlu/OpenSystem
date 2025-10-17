/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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
#include "RenderFrameBase.h"

#include "HTMLFrameElementBase.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderView.h"
#include "ScriptDisallowedScope.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderFrameBase);

RenderFrameBase::RenderFrameBase(Type type, HTMLFrameElementBase& element, RenderStyle&& style)
    : RenderWidget(type, element, WTFMove(style))
{
}

RenderFrameBase::~RenderFrameBase() = default;

inline bool shouldExpandFrame(LayoutUnit width, LayoutUnit height, bool hasFixedWidth, bool hasFixedHeight)
{
    // If the size computed to zero never expand.
    if (!width || !height)
        return false;
    // Really small fixed size frames can't be meant to be scrolled and are there probably by mistake. Avoid expanding.
    static const unsigned smallestUsefullyScrollableDimension = 8;
    if (hasFixedWidth && width < LayoutUnit(smallestUsefullyScrollableDimension))
        return false;
    if (hasFixedHeight && height < LayoutUnit(smallestUsefullyScrollableDimension))
        return false;
    return true;
}

}
