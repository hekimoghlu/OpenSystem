/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#include "CanvasFilterContextSwitcher.h"

#include "CanvasLayerContextSwitcher.h"
#include "CanvasRenderingContext2DBase.h"
#include "FloatRect.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CanvasFilterContextSwitcher);

std::unique_ptr<CanvasFilterContextSwitcher> CanvasFilterContextSwitcher::create(CanvasRenderingContext2DBase& context, const FloatRect& bounds)
{
    if (context.state().filterOperations.isEmpty())
        return nullptr;

    auto filter = context.createFilter(bounds);
    if (!filter)
        return nullptr;

    auto filterSwitcher = makeUnique<CanvasFilterContextSwitcher>(context);
    if (!filterSwitcher)
        return nullptr;

    auto targetSwitcher = CanvasLayerContextSwitcher::create(context, bounds, WTFMove(filter));
    if (!targetSwitcher)
        return nullptr;

    context.modifiableState().targetSwitcher = WTFMove(targetSwitcher);
    return filterSwitcher;
}

CanvasFilterContextSwitcher::CanvasFilterContextSwitcher(CanvasRenderingContext2DBase& context)
    : m_context(context)
{
    context.save();
    context.realizeSaves();
}

CanvasFilterContextSwitcher::~CanvasFilterContextSwitcher()
{
    protectedContext()->restore();
}

FloatRect CanvasFilterContextSwitcher::expandedBounds() const
{
    return m_context->state().targetSwitcher->expandedBounds();
}

} // namespace WebCore
