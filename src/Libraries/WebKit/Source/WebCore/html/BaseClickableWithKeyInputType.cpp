/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "BaseClickableWithKeyInputType.h"

#include "HTMLInputElement.h"
#include "KeyboardEvent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BaseClickableWithKeyInputType);

using namespace HTMLNames;

auto BaseClickableWithKeyInputType::handleKeydownEvent(HTMLInputElement& element, KeyboardEvent& event) -> ShouldCallBaseEventHandler
{
    const String& key = event.keyIdentifier();
    if (key == "U+0020"_s) {
        element.setActive(true);
        // No setDefaultHandled(), because IE dispatches a keypress in this case
        // and the caller will only dispatch a keypress if we don't call setDefaultHandled().
        return ShouldCallBaseEventHandler::No;
    }
    return ShouldCallBaseEventHandler::Yes;
}

void BaseClickableWithKeyInputType::handleKeypressEvent(HTMLInputElement& element, KeyboardEvent& event)
{
    int charCode = event.charCode();
    if (charCode == '\r') {
        element.dispatchSimulatedClick(&event);
        event.setDefaultHandled();
        return;
    }
    if (charCode == ' ') {
        // Prevent scrolling down the page.
        event.setDefaultHandled();
    }
}

void BaseClickableWithKeyInputType::handleKeyupEvent(InputType& inputType, KeyboardEvent& event)
{
    const String& key = event.keyIdentifier();
    if (key != "U+0020"_s)
        return;
    // Simulate mouse click for spacebar for button types.
    inputType.dispatchSimulatedClickIfActive(event);
}

// FIXME: Could share this with BaseCheckableInputType and RangeInputType if we had a common base class.
bool BaseClickableWithKeyInputType::accessKeyAction(HTMLInputElement& element, bool sendMouseEvents)
{
    return element.dispatchSimulatedClick(0, sendMouseEvents ? SendMouseUpDownEvents : SendNoEvents);
}

auto BaseClickableWithKeyInputType::handleKeydownEvent(KeyboardEvent& event) -> ShouldCallBaseEventHandler
{
    ASSERT(element());
    return handleKeydownEvent(*element(), event);
}

void BaseClickableWithKeyInputType::handleKeypressEvent(KeyboardEvent& event)
{
    ASSERT(element());
    handleKeypressEvent(*element(), event);
}

void BaseClickableWithKeyInputType::handleKeyupEvent(KeyboardEvent& event)
{
    handleKeyupEvent(*this, event);
}

bool BaseClickableWithKeyInputType::accessKeyAction(bool sendMouseEvents)
{
    InputType::accessKeyAction(sendMouseEvents);
    ASSERT(element());
    return accessKeyAction(*element(), sendMouseEvents);
}

} // namespace WebCore
