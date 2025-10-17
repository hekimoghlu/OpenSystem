/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#include "EventHandler.h"

#include "HandleUserInputEventResult.h"
#include "MouseEventWithHitTestResults.h"

namespace WebCore {

HandleUserInputEventResult EventHandler::passMouseMoveEventToSubframe(MouseEventWithHitTestResults& mouseEventAndResult, LocalFrame& subframe, HitTestResult* hitTestResult)
{
#if ENABLE(DRAG_SUPPORT)
    if (m_mouseDownMayStartDrag && !m_mouseDownWasInSubframe)
        return false;
#endif

    subframe.eventHandler().handleMouseMoveEvent(mouseEventAndResult.event(), hitTestResult);
    return true;
}

bool EventHandler::eventActivatedView(const PlatformMouseEvent& event) const
{
    return event.didActivateWebView();
}

}
