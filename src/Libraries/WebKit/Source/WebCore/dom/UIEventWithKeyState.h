/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#pragma once

#include "EventModifierInit.h"
#include "PlatformEvent.h"
#include "UIEvent.h"

namespace WebCore {

class UIEventWithKeyState : public UIEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(UIEventWithKeyState);
public:
    using Modifier = PlatformEvent::Modifier;

    bool ctrlKey() const { return m_modifiers.contains(Modifier::ControlKey); }
    bool shiftKey() const { return m_modifiers.contains(Modifier::ShiftKey); }
    bool altKey() const { return m_modifiers.contains(Modifier::AltKey); }
    bool metaKey() const { return m_modifiers.contains(Modifier::MetaKey); }
    bool capsLockKey() const { return m_modifiers.contains(Modifier::CapsLockKey); }

    OptionSet<Modifier> modifierKeys() const { return m_modifiers; }

    WEBCORE_EXPORT bool getModifierState(const String& keyIdentifier) const;

protected:
    UIEventWithKeyState(enum EventInterfaceType eventInterface)
        : UIEvent(eventInterface)
    {
    }

    UIEventWithKeyState(enum EventInterfaceType eventInterface, const AtomString& type, CanBubble canBubble, IsCancelable cancelable, IsComposed isComposed,
        RefPtr<WindowProxy>&& view, int detail, OptionSet<Modifier> modifiers)
        : UIEvent(eventInterface, type, canBubble, cancelable, isComposed, WTFMove(view), detail)
        , m_modifiers(modifiers)
    {
    }

    UIEventWithKeyState(enum EventInterfaceType eventInterface, const AtomString& type, CanBubble canBubble, IsCancelable cancelable, IsComposed isComposed,
        MonotonicTime timestamp, RefPtr<WindowProxy>&& view, int detail, OptionSet<Modifier> modifiers, IsTrusted isTrusted)
        : UIEvent(eventInterface, type, canBubble, cancelable, isComposed, timestamp, WTFMove(view), detail, isTrusted)
        , m_modifiers(modifiers)
    {
    }

    UIEventWithKeyState(enum EventInterfaceType eventInterface, const AtomString& type, const EventModifierInit& initializer, IsTrusted isTrusted = IsTrusted::No)
        : UIEvent(eventInterface, type, initializer, isTrusted)
        , m_modifiers(modifiersFromInitializer(initializer))
    {
    }

    void setModifierKeys(bool ctrlKey, bool altKey, bool shiftKey, bool metaKey);

private:
    OptionSet<Modifier> m_modifiers;

    static OptionSet<Modifier> modifiersFromInitializer(const EventModifierInit& initializer);
};

UIEventWithKeyState* findEventWithKeyState(Event*);

} // namespace WebCore
