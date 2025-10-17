/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#include "ScrollbarsControllerMock.h"

#include "ScrollableArea.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollbarsControllerMock);

ScrollbarsControllerMock::ScrollbarsControllerMock(ScrollableArea& scrollableArea, Function<void(const String&)>&& logger)
    : ScrollbarsController(scrollableArea)
    , m_logger(WTFMove(logger))
{
}

ScrollbarsControllerMock::~ScrollbarsControllerMock() = default;

void ScrollbarsControllerMock::didAddVerticalScrollbar(Scrollbar* scrollbar)
{
    m_verticalScrollbar = scrollbar;
    ScrollbarsController::didAddVerticalScrollbar(scrollbar);
}

void ScrollbarsControllerMock::didAddHorizontalScrollbar(Scrollbar* scrollbar)
{
    m_horizontalScrollbar = scrollbar;
    ScrollbarsController::didAddHorizontalScrollbar(scrollbar);
}

void ScrollbarsControllerMock::willRemoveVerticalScrollbar(Scrollbar* scrollbar)
{
    ScrollbarsController::willRemoveVerticalScrollbar(scrollbar);
    m_verticalScrollbar = nullptr;
}

void ScrollbarsControllerMock::willRemoveHorizontalScrollbar(Scrollbar* scrollbar)
{
    ScrollbarsController::willRemoveHorizontalScrollbar(scrollbar);
    m_horizontalScrollbar = nullptr;
}

void ScrollbarsControllerMock::mouseEnteredContentArea()
{
    m_logger("mouseEnteredContentArea"_s);
    ScrollbarsController::mouseEnteredContentArea();
}

void ScrollbarsControllerMock::mouseMovedInContentArea()
{
    m_logger("mouseMovedInContentArea"_s);
    ScrollbarsController::mouseMovedInContentArea();
}

void ScrollbarsControllerMock::mouseExitedContentArea()
{
    m_logger("mouseExitedContentArea"_s);
    ScrollbarsController::mouseExitedContentArea();
}

ASCIILiteral ScrollbarsControllerMock::scrollbarPrefix(Scrollbar* scrollbar) const
{
    return scrollbar == m_verticalScrollbar ? "Vertical"_s : scrollbar == m_horizontalScrollbar ? "Horizontal"_s : "Unknown"_s;
}

void ScrollbarsControllerMock::mouseEnteredScrollbar(Scrollbar* scrollbar) const
{
    m_logger(makeString("mouseEntered"_s, scrollbarPrefix(scrollbar), "Scrollbar"_s));
    ScrollbarsController::mouseEnteredScrollbar(scrollbar);
}

void ScrollbarsControllerMock::mouseExitedScrollbar(Scrollbar* scrollbar) const
{
    m_logger(makeString("mouseExited"_s, scrollbarPrefix(scrollbar), "Scrollbar"_s));
    ScrollbarsController::mouseExitedScrollbar(scrollbar);
}

void ScrollbarsControllerMock::mouseIsDownInScrollbar(Scrollbar* scrollbar, bool isPressed) const
{
    m_logger(makeString(isPressed ? "mouseIsDownIn"_s : "mouseIsUpIn"_s, scrollbarPrefix(scrollbar), "Scrollbar"_s));
    ScrollbarsController::mouseIsDownInScrollbar(scrollbar, isPressed);
}

} // namespace WebCore
