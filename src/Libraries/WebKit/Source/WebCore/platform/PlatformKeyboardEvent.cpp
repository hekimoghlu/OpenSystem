/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#include "PlatformKeyboardEvent.h"

#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PlatformKeyboardEvent);

std::optional<OptionSet<PlatformEvent::Modifier>> PlatformKeyboardEvent::s_currentModifiers;    

bool PlatformKeyboardEvent::currentCapsLockState()
{
    return currentStateOfModifierKeys().contains(PlatformEvent::Modifier::CapsLockKey);
}

void PlatformKeyboardEvent::getCurrentModifierState(bool& shiftKey, bool& ctrlKey, bool& altKey, bool& metaKey)
{
    auto currentModifiers = currentStateOfModifierKeys();
    shiftKey = currentModifiers.contains(PlatformEvent::Modifier::ShiftKey);
    ctrlKey = currentModifiers.contains(PlatformEvent::Modifier::ControlKey);
    altKey = currentModifiers.contains(PlatformEvent::Modifier::AltKey);
    metaKey = currentModifiers.contains(PlatformEvent::Modifier::MetaKey);
}

void PlatformKeyboardEvent::setCurrentModifierState(OptionSet<Modifier> modifiers)
{
    ASSERT(isMainThread());
    s_currentModifiers = modifiers;
}

}
