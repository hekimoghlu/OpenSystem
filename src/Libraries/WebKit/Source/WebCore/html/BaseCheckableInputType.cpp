/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
#include "BaseCheckableInputType.h"

#include "CommonAtomStrings.h"
#include "DOMFormData.h"
#include "FormController.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "KeyboardEvent.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BaseCheckableInputType);

using namespace HTMLNames;

FormControlState BaseCheckableInputType::saveFormControlState() const
{
    ASSERT(element());
    return { element()->checked() ? onAtom() : offAtom() };
}

void BaseCheckableInputType::restoreFormControlState(const FormControlState& state)
{
    ASSERT(element());
    element()->setChecked(state[0] == onAtom());
}

bool BaseCheckableInputType::appendFormData(DOMFormData& formData) const
{
    ASSERT(element());
    if (!element()->checked())
        return false;
    formData.append(element()->name(), element()->value());
    return true;
}

auto BaseCheckableInputType::handleKeydownEvent(KeyboardEvent& event) -> ShouldCallBaseEventHandler
{
    const String& key = event.keyIdentifier();
    if (key == "U+0020"_s) {
        ASSERT(element());
        element()->setActive(true);
        // No setDefaultHandled(), because IE dispatches a keypress in this case
        // and the caller will only dispatch a keypress if we don't call setDefaultHandled().
        return ShouldCallBaseEventHandler::No;
    }
    return ShouldCallBaseEventHandler::Yes;
}

void BaseCheckableInputType::handleKeypressEvent(KeyboardEvent& event)
{
    if (event.charCode() == ' ') {
        // Prevent scrolling down the page.
        event.setDefaultHandled();
    }
}

bool BaseCheckableInputType::canSetStringValue() const
{
    return false;
}

// FIXME: Could share this with BaseClickableWithKeyInputType and RangeInputType if we had a common base class.
bool BaseCheckableInputType::accessKeyAction(bool sendMouseEvents)
{
    ASSERT(element());
    return InputType::accessKeyAction(sendMouseEvents) || protectedElement()->dispatchSimulatedClick(0, sendMouseEvents ? SendMouseUpDownEvents : SendNoEvents);
}

String BaseCheckableInputType::fallbackValue() const
{
    return onAtom();
}

bool BaseCheckableInputType::storesValueSeparateFromAttribute()
{
    return false;
}

void BaseCheckableInputType::setValue(const String& sanitizedValue, bool, TextFieldEventBehavior, TextControlSetValueSelection)
{
    ASSERT(element());
    element()->setAttributeWithoutSynchronization(valueAttr, AtomString { sanitizedValue });
}

void BaseCheckableInputType::fireInputAndChangeEvents()
{
    if (!element()->isConnected())
        return;

    if (!shouldSendChangeEventAfterCheckedChanged())
        return;

    Ref protectedThis { *this };
    element()->setTextAsOfLastFormControlChangeEvent(String());
    element()->dispatchInputEvent();
    if (auto* element = this->element())
        element->dispatchFormControlChangeEvent();
}

} // namespace WebCore
