/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
#include "CaretAnimator.h"

#include "GraphicsContext.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CaretAnimator);

bool CaretAnimator::isBlinkingSuspended() const
{
#if ENABLE(ACCESSIBILITY_NON_BLINKING_CURSOR)
    if (m_prefersNonBlinkingCursor)
        return true;
#endif
    return m_isBlinkingSuspended;
}

Page* CaretAnimator::page() const
{
    if (auto* document = m_client.document())
        return document->page();
    
    return nullptr;
}

void CaretAnimator::stop(CaretAnimatorStopReason)
{
    if (!m_isActive)
        return;

    didEnd();
}

void CaretAnimator::serviceCaretAnimation()
{
    if (!isActive())
        return;

    updateAnimationProperties();
}

void CaretAnimator::scheduleAnimation()
{
    if (RefPtr page = this->page())
        page->scheduleRenderingUpdate(RenderingUpdateStep::CaretAnimation);
}

void CaretAnimator::paint(GraphicsContext& context, const FloatRect& caret, const Color& color, const LayoutPoint&) const
{
    context.fillRect(caret, color);
}

LayoutRect CaretAnimator::caretRepaintRectForLocalRect(LayoutRect rect) const
{
    return rect;
}

} // namespace WebCore
