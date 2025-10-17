/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include "ScrollbarsController.h"

#include "ScrollableArea.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

#if !PLATFORM(MAC) && !PLATFORM(WPE) && !PLATFORM(GTK)

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollbarsController);

std::unique_ptr<ScrollbarsController> ScrollbarsController::create(ScrollableArea& scrollableArea)
{
    return makeUnique<ScrollbarsController>(scrollableArea);
}
#endif

ScrollbarsController::ScrollbarsController(ScrollableArea& scrollableArea)
    : m_scrollableArea(scrollableArea)
{
}

bool ScrollbarsController::shouldSuspendScrollbarAnimations() const
{
    return scrollableArea().shouldSuspendScrollAnimations();
}

void ScrollbarsController::cancelAnimations()
{
    setScrollbarAnimationsUnsuspendedByUserInteraction(false);
}

void ScrollbarsController::didBeginScrollGesture()
{
    setScrollbarAnimationsUnsuspendedByUserInteraction(true);
}

void ScrollbarsController::didEndScrollGesture()
{
    setScrollbarAnimationsUnsuspendedByUserInteraction(true);
}

void ScrollbarsController::mayBeginScrollGesture()
{
    setScrollbarAnimationsUnsuspendedByUserInteraction(true);
}

void ScrollbarsController::updateScrollbarsThickness()
{
    if (auto verticalScrollbar = scrollableArea().verticalScrollbar())
        verticalScrollbar->updateScrollbarThickness();

    if (auto horizontalScrollbar = scrollableArea().horizontalScrollbar())
        horizontalScrollbar->updateScrollbarThickness();
}

} // namespace WebCore
