/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#include "UIEventWithKeyState.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(UIEventWithKeyState);

auto UIEventWithKeyState::modifiersFromInitializer(const EventModifierInit& initializer) -> OptionSet<Modifier>
{
    OptionSet<Modifier> result;
    if (initializer.ctrlKey)
        result.add(Modifier::ControlKey);
    if (initializer.altKey)
        result.add(Modifier::AltKey);
    if (initializer.shiftKey)
        result.add(Modifier::ShiftKey);
    if (initializer.metaKey)
        result.add(Modifier::MetaKey);
    if (initializer.modifierAltGraph)
        result.add(Modifier::AltGraphKey);
    if (initializer.modifierCapsLock)
        result.add(Modifier::CapsLockKey);
    return result;
}

bool UIEventWithKeyState::getModifierState(const String& keyIdentifier) const
{
    if (keyIdentifier == "Control"_s)
        return ctrlKey();
    if (keyIdentifier == "Shift"_s)
        return shiftKey();
    if (keyIdentifier == "Alt"_s)
        return altKey();
    if (keyIdentifier == "Meta"_s)
        return metaKey();
    if (keyIdentifier == "CapsLock"_s)
        return capsLockKey();
    // FIXME: The specification also has Fn, FnLock, Hyper, NumLock, Super, ScrollLock, Symbol, SymbolLock.
    return false;
}

void UIEventWithKeyState::setModifierKeys(bool ctrlKey, bool altKey, bool shiftKey, bool metaKey)
{
    OptionSet<Modifier> result;
    if (ctrlKey)
        result.add(Modifier::ControlKey);
    if (altKey)
        result.add(Modifier::AltKey);
    if (shiftKey)
        result.add(Modifier::ShiftKey);
    if (metaKey)
        result.add(Modifier::MetaKey);
    m_modifiers = result;
}

UIEventWithKeyState* findEventWithKeyState(Event* event)
{
    for (Event* e = event; e; e = e->underlyingEvent())
        if (e->isKeyboardEvent() || e->isMouseEvent())
            return static_cast<UIEventWithKeyState*>(e);
    return 0;
}

} // namespace WebCore
